# mmdetection
for 2021tianchi


可以调整的超参:
1.anchor,印象里mmdetection是高宽比;如果担心出错可以对称设置anchor,如[0.2, 0.5, 1, 2, 5];
2.frozen_stages, 默认是1, 印象里有个大佬说1一般比较好,个人感觉可能是因为第一层都是边缘之类的特征,对所有物体都是通用的;可以尝试下-1(所有层都更新参数);




训练前要注意的参数:
1.学习率,根据batchsize;
2.预训练模型,load_from改一下;
3.# fp16 settings
fp16 = dict(loss_scale=512.)
4.配置文件的work_dir改一下


训练:(注意添加预训练模型,在default_runtime.py配置文件中添加load_from='xxxxxx.pth'
python tools/train.py configs/tile/faster_rcnn_r50_fpn_1x_tile.py
python tools/train.py configs/tile/cascade_rcnn_r50_fpn_1x_coco.py
python tools/train.py configs/tile/detectors_cascade_rcnn_r50_1x_coco.py

验证:
python tools/test.py configs/tile/faster_rcnn_r50_fpn_1x_tile.py work_dirs/faster_rcnn_r50_fpn_1x_tile/epoch_10.pth --eval bbox
python tools/test.py configs/tile/cascade_rcnn_r50_fpn_1x_coco.py work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_11.pth --eval bbox

预测:
为了使用mmdetection自带的多进程预测,先使用tools中的generate_test_json.py为测试集生成伪标签,路径填入配置文件的test_pipeline中;
python tools/test.py configs/tile/faster_rcnn_r50_fpn_1x_tile.py work_dirs/faster_rcnn_r50_fpn_1x_tile/epoch_10.pth --json_out faster_r50
python tools/test.py configs/tile/cascade_rcnn_r50_fpn_1x_coco.py work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_11.pth --json_out cas_r50
python tools/test.py configs/tile/detectors_cascade_rcnn_r50_1x_coco.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_12.pth --json_out DetectoRS_r50

常见错误：
1、RuntimeError: DataLoader worker (pid 49) is killed by signal: Terminated.
和 ValueError: need at least one array to concatenate
解决：https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html
自注：注意 mmdet/dataset/coco.py 里面的类别名称要和json里面对上！！这里如果json是中文，类别名也要写成中文，不能随便写 CLASS = ['0' '1' '2' '3' '4' '5' ]


2、第一个epoch预测完报错：
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 1519/1519, 15.3 task/s, elapsed: 99s, ETA:     0sTraceback (most recent call last):
  File "tools/test.py", line 231, in <module>
    main()
  File "tools/test.py", line 217, in main
    print(dataset.evaluate(outputs, **eval_kwargs))
  File "/home/user/mmdetection/mmdet/datasets/coco.py", line 416, in evaluate
    result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
  File "/home/user/mmdetection/mmdet/datasets/coco.py", line 361, in format_results
    result_files = self.results2json(results, jsonfile_prefix)
  File "/home/user/mmdetection/mmdet/datasets/coco.py", line 293, in results2json
    json_results = self._det2json(results)
  File "/home/user/mmdetection/mmdet/datasets/coco.py", line 230, in _det2json
    data['category_id'] = self.cat_ids[label]
IndexError: list index out of range

原因：config里面num_classes没改过来，还是原来的80；



3、预测(这里是原图预测;如果是切图预测,要把Pad去掉)
不要直接去掉configs里面的Resize操作，否则报错
KeyError: 'scale_factor'
改成下面这样就可以了：
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),  # clw delete
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



其他说明:
1.配置文件中,style='pytorch',表示网络风格,pytorch的stride=2的层是conv3x3,而caffe是conv1x1;
2.nms用的multiclass_nms,默认是类别相关的(class_agnostic=False);改成类别无关的往往掉点(因为找回率降低了);



