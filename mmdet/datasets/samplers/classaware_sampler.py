import math
import pickle as pk

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler


class ClassAwareSampler(Sampler):
    def __init__(self, dataset, class_sample_path=None):

        self.dataset = dataset
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples

        with open(class_sample_path, "rb") as f:
            self.class_dic = pk.load(f)
        self.class_num = len(self.class_dic.keys())
        # self.class_num_list = [
        #     self.class_dic[i+1] for i in range(self.class_num)
        # ]
        self.class_num_list = [
            len(self.class_dic[i]) for i in range(self.class_num) # clw note: classdict的每个元素对应的值是一个列表,指出了哪些图(id)有这个类的gt
        ]

        self.indices = None

    def __iter__(self):
        print('clw: call __iter__ in ClassAwareSampler()')

        def gen_class_num_indices(class_num_list):
            class_indices = np.random.permutation(self.class_num)
            # id_indices = [
            #     self.class_dic[class_indice + 1][
            #         np.random.permutation(class_num_list[class_indice])[0]
            #     ] for class_indice in class_indices if class_num_list[class_indice] != 0
            # ]
            id_indices = []
            for class_indice in class_indices:
                if class_num_list[class_indice] != 0:
                    aaa = np.random.permutation(class_num_list[class_indice])[0]
                    bbb = self.class_dic[class_indice][aaa]
                    id_indices.append( bbb )
            return id_indices

        # deterministically shuffle based on epoch
        np.random.seed(self.epoch + 1)
        num_bins = int(math.floor(self.total_size * 1.0 / self.class_num)) # 3228 samples, 8 classes  num_bins=403
        indices = []
        for i in range(num_bins):
            indices += gen_class_num_indices(self.class_num_list)

        #ccc = set(indices)  # clw note: about 1700+
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]  # origin indices is 3224, from 3224 to 3228
        assert len(indices) == self.total_size  # 3228
        # subsample
        # offset = self.num_samples  # clw note: why ?  TODO
        # indices = indices[offset : offset + self.num_samples]  # 3228
        # assert len(indices) == self.num_samples
        self.indices = indices
        self.epoch += 1  # clw modify
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        print('clw: call set_epoch()')  # clw note: never call it...
        self.epoch = epoch

