import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image
import glob


class MallDataset(Dataset):

    def __init__(self, preprocess, sets, is_map=False):
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

        if is_map:
            pass  # TODO: include density map import
        else:
            gt_paths = []
            for s in sets:
                path = fr"{root_path}\{s}\train\gt\*.jpg"
                gt_paths += glob.glob(path)

            gts = np.zeros((samples, 1))
            for i, f in enumerate(gt_paths):
                data = np.genfromtxt(f, delimiter=',')
                count = np.sum(data)
                gts[i, 0] = count

        self.y = torch.from_numpy(gts)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples

