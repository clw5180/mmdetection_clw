import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

###############################################
from mmdet.apis import inference_detector, init_detector  # clw add
import numpy as np
import torchvision
###########################################

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    print('clw: using single_gpu_test() !!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            ########### clw note: for debug
            # for idx, item in enumerate(result[0]):
            #     for row in item:
            #         print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
            #         if row[2] - row[0] == 0 or row[3] - row[1] == 0:
            #             print('aaaa')
            #########

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

# clw add
def single_gpu_test_crop_img(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    print('clw: using single_gpu_test_crop_img() !!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):  # data['img][0]: tensor (1, 3, 6016, 8192)
        img_h = data['img'][0].shape[2]
        img_w = data['img'][0].shape[3]
        with torch.no_grad():
            # 如果是 4096x3500,直接原图预测
            if img_h <= 3500 and img_w <= 4096:
            #if img_h <= 10000 and img_w <= 10000:
                result = model(return_loss=False, rescale=True, **data)
            else:
                # 否则切图, 4 nums
                ##############################
                overlap_h = 272
                overlap_w = 256

                crop_h = round((img_h + overlap_h) / 2)   # clw note: the size can be divided by 32 is better
                crop_w = round((img_w + overlap_w) / 2)
                #crop_h = 800
                #crop_w = 1344
                # crop_w = 1333

                #step_h = int(0.8 * crop_h)
                #step_w = int(0.8 * crop_w)
                step_h = crop_h - overlap_h
                step_w = crop_w - overlap_w

                nms_iou_thr = model.module.test_cfg['rcnn']['nms']['iou_threshold']
                results_crop = [[] for _ in range(len(model.module.CLASSES))]
                data['img_metas'][0].data[0][0]['ori_shape'] = (crop_h, crop_w)
                data['img_metas'][0].data[0][0]['img_shape'] = (crop_h, crop_w)
                data['img_metas'][0].data[0][0]['pad_shape'] = (crop_h, crop_w)
                img_tensor_orig = data['img'][0].clone()
                for start_h in range(0, img_h-crop_h+1, step_h):  # imgsz is crop step here,
                    if start_h + crop_h > img_h:  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                        start_h = img_h - crop_h

                    for start_w in range(0, img_w-crop_w+1, step_w):
                        if start_w + crop_w > img_w:  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                            start_w = img_w - crop_w
                        # crop
                        print(start_h, start_w)
                        data['img'][0] = img_tensor_orig[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

                        result = model(return_loss=False, rescale=True, **data)  # result[0]: model.module.CLASSES 个list,每个里面装着(n, 5) ndarray
                        #result = model(return_loss=False, rescale=False, **data)  # clw modify
                        for idx, item in enumerate(result[0]):
                            for row in item:
                                #print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
                                if row[2] - row[0] == 0 or row[3] - row[1] == 0:
                                    print('===================================================================')
                                    continue
                                row[[0, 2]] += start_w
                                row[[1, 3]] += start_h
                                results_crop[idx].append(row)

                results_afternms = []
                for idx, res in enumerate(results_crop):
                    if len(res) == 0:
                        results_afternms.append(np.array([])) # clw note: it's really important!!
                        continue
                    else:
                        prediction = torch.tensor(res)
                        boxes, scores = prediction[:, :4], prediction[:, 4]  # boxes (offset by class), scores
                        i = torchvision.ops.boxes.nms(boxes, scores, nms_iou_thr)
                        results_afternms.append(prediction[i].numpy())
                result = [ results_afternms ]
                ##############################

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


# clw added
import cv2
def single_gpu_test_rotate_rect_img(model,
                            data_loader,
                            show=False,
                            out_dir=None,
                            show_score_thr=0.3):
    print('clw: using single_gpu_test_rotate_rect_img() !!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            ########### clw note: for debug
            # for idx, item in enumerate(result[0]):
            #     if item.size == 0:
            #         print('111')

            #    for row in item:
            #         print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
            #         if row[2] - row[0] == 0 or row[3] - row[1] == 0:
            #             print('aaaa')
            #########

        ##
        img_name = data['img_metas'][0].data[0][0]['ori_filename']
        # origin_name = img_name.split('CAM')[0] + 'CAM' + img_name.split('CAM')[1][0] + '.jpg'
        # data['img_metas'][0].data[0][0]['ori_filename'] = origin_name
        # data['img_metas'][0].data[0][0]['filename'] = data['img_metas'][0].data[0][0]['filename'].rsplit('/', 1)[0] + '/' + origin_name

        aaa = img_name[:-4].split('_')[-9:]
        bbb = [float(a) for a in aaa]
        M_perspective_inv = np.array(bbb).reshape(3, 3)

        for i in range(len(result[0])):
            ddd = []
            ccc = result[0][i][:, :4]  # (n, 4)
            if ccc.size == 0:
                continue
            for xyxy in ccc:
                x1 = xyxy[0]
                y1 = xyxy[1]
                x2 = xyxy[2]
                y2 = xyxy[3]
                cnt = np.array( ((x1, y1), (x1, y2), (x2, y2), (x2, y1)))
                ddd.append(cnt)
            ddd = np.array(ddd)

            #
            fff = []
            src_pts = cv2.perspectiveTransform(ddd, M_perspective_inv)
            for cnt in src_pts:
                rect = cv2.boundingRect(cnt)
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[0] + rect[2]
                y2 = rect[1] + rect[3]
                ggg = np.array( (x1, y1, x2, y2 ) )
                fff.append(ggg)
            fff = np.array(fff)

            result[0][i][:, :4] = fff  # result[0][i] = np.concatenate((fff, result[0][i][:, 4]), axis=1)
        ##

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


# clw added
def single_gpu_test_processed_rect_img(model,
                            data_loader,
                            show=False,
                            out_dir=None,
                            show_score_thr=0.3):
    print('clw: using single_gpu_test_processed_rect_img() !!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            ########### clw note: for debug
            # for idx, item in enumerate(result[0]):
            #     if item.size == 0:
            #         print('111')

            #    for row in item:
            #         print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
            #         if row[2] - row[0] == 0 or row[3] - row[1] == 0:
            #             print('aaaa')
            #########

        ##
        img_name = data['img_metas'][0].data[0][0]['ori_filename']


        aaa = img_name[:-4].split('_')[-2:]
        x_rect_left = int(aaa[0])
        y_rect_up = int(aaa[1])

        for i in range(len(result[0])):
            ddd = []
            ccc = result[0][i][:, :4]  # (n, 4)
            if ccc.size == 0:
                continue
            for xyxy in ccc:
                x1 = xyxy[0] + x_rect_left
                y1 = xyxy[1] + y_rect_up
                x2 = xyxy[2] + x_rect_left
                y2 = xyxy[3] + y_rect_up
                cnt = np.array( (x1, y1, x2, y2))
                ddd.append(cnt)
            ddd = np.array(ddd)

            result[0][i][:, :4] = ddd  # result[0][i] = np.concatenate((fff, result[0][i][:, 4]), axis=1)
        ##

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


# clw added
def single_gpu_test_processed_rect_crop_img(model,
                            data_loader,
                            show=False,
                            out_dir=None,
                            show_score_thr=0.3):
    print('clw: using single_gpu_test_processed_rect_crop_img() !!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        print(data['img_metas'][0].data[0][0]['ori_filename'])
        img_h = data['img'][0].shape[2]
        img_w = data['img'][0].shape[3]
        with torch.no_grad():
            # 否则切图, 4 nums
            ##############################
            overlap_h = 128
            overlap_w = 128
            crop_h = 800
            crop_w = 1333

            step_h = crop_h - overlap_h
            step_w = crop_w - overlap_w

            nms_iou_thr = model.module.test_cfg['rcnn']['nms']['iou_threshold']
            results_crop = [[] for _ in range(len(model.module.CLASSES))]
            data['img_metas'][0].data[0][0]['ori_shape'] = (crop_h, crop_w)
            data['img_metas'][0].data[0][0]['img_shape'] = (crop_h, crop_w)
            data['img_metas'][0].data[0][0]['pad_shape'] = (crop_h, crop_w)
            img_tensor_orig = data['img'][0].clone()
            for start_h in range(0, img_h-crop_h+1, step_h):  # imgsz is crop step here,
                if start_h + crop_h > img_h:  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                    start_h = img_h - crop_h

                for start_w in range(0, img_w-crop_w+1, step_w):
                    if start_w + crop_w > img_w:  # 如果最后剩下的不到imgsz,则step少一些,保证切的图尺寸不变
                        start_w = img_w - crop_w
                    # crop
                    print(start_h, start_w)
                    data['img'][0] = img_tensor_orig[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

                    result = model(return_loss=False, rescale=True, **data)  # result[0]: model.module.CLASSES 个list,每个里面装着(n, 5) ndarray
                    #result = model(return_loss=False, rescale=False, **data)  # clw modify
                    for idx, item in enumerate(result[0]):
                        for row in item:
                            #print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
                            if row[2] - row[0] == 0 or row[3] - row[1] == 0:
                                print('===================================================================')
                                continue
                            row[[0, 2]] += start_w
                            row[[1, 3]] += start_h
                            results_crop[idx].append(row)

            results_afternms = []
            for idx, res in enumerate(results_crop):
                if len(res) == 0:
                    results_afternms.append(np.array([])) # clw note: it's really important!!
                    continue
                else:
                    prediction = torch.tensor(res)
                    boxes, scores = prediction[:, :4], prediction[:, 4]  # boxes (offset by class), scores
                    i = torchvision.ops.boxes.nms(boxes, scores, nms_iou_thr)
                    results_afternms.append(prediction[i].numpy())
            result = [ results_afternms ]
            ##############################





            ########### clw note: for debug
            # for idx, item in enumerate(result[0]):
            #     if item.size == 0:
            #         print('111')

            #    for row in item:
            #         print('boxw:', row[2] - row[0],  'boxh:', row[3] - row[1] )
            #         if row[2] - row[0] == 0 or row[3] - row[1] == 0:
            #             print('aaaa')
            #########

        ##
        img_name = data['img_metas'][0].data[0][0]['ori_filename']


        aaa = img_name[:-4].split('_')[-2:]
        x_rect_left = int(aaa[0])
        y_rect_up = int(aaa[1])

        for i in range(len(result[0])):
            ddd = []
            if result[0][i].size == 0:
                continue
            ccc = result[0][i][:, :4]  # (n, 4)
            for xyxy in ccc:
                x1 = xyxy[0] + x_rect_left
                y1 = xyxy[1] + y_rect_up
                x2 = xyxy[2] + x_rect_left
                y2 = xyxy[3] + y_rect_up
                cnt = np.array( (x1, y1, x2, y2))
                ddd.append(cnt)
            ddd = np.array(ddd)

            result[0][i][:, :4] = ddd  # result[0][i] = np.concatenate((fff, result[0][i][:, 4]), axis=1)
        ##

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
