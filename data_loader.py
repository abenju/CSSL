import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image
import glob
import cv2


class MallDataset(Dataset):

    def __init__(self, preprocess, sets, is_map=False, new_size=None):
        root_path = os.getenv("DATA_ROOT")

        input_paths = []
        for s in sets:
            path = fr"{root_path}\{s}\train\input\*.jpg"
            input_paths += glob.glob(path)

        imgs = []
        for path in input_paths:
            image = Image.open(path)
            image = preprocess(image)
            imgs.append(image)

        samples = len(imgs)
        imgs = torch.stack(imgs)

        self.x = imgs
        self.n_samples = samples

        gt_paths = []
        for s in sets:
            path = fr"{root_path}\{s}\train\gt\*.csv"
            gt_paths += glob.glob(path)

        data_ps = []
        for f in gt_paths:
            data_ps.append(np.genfromtxt(f, delimiter=','))

        if is_map:
            gts = []
            for i, p in enumerate(data_ps):
                gt_size = p.shape
                gt = cv2.resize(p, (new_size[1], new_size[0]))
                gt = gt * ((gt_size[0] * gt_size[1]) / (new_size[0] * new_size[1]))
                gts.append(np.expand_dims(gt, axis=0))
            gts = torch.Tensor(np.array(gts))

        else:
            gts = np.zeros((samples, 1))
            for i, p in enumerate(data_ps):
                count = np.sum(p)
                gts[i, 0] = count
            gts = torch.from_numpy(gts)

        self.y = gts

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples

