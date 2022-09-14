import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image


class MallDataSet(Dataset):
    def __init__(self):
        samples = 0
        feature_size = 512
        gt_size = 480 * 640  # (479, 640)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        csv_path = r"D:\Documents\mscProject\aduen\counts.csv"

        # features_dir = r'D:\Documents\mscProject\aduen\data\features\Feature_files'
        # features_files = next(os.walk(features_dir))[2]
        # features_paths = [os.path.join(features_dir, x) for x in features_files]

        input_dir = r'D:\Documents\mscProject\IJCAI_2021_Continual_Crowd_Counting_Challenge\data\mall\train\input'
        input_files = next(os.walk(input_dir))[2]
        input_paths = [os.path.join(input_dir, x) for x in input_files]

        imgs = []
        for path in input_paths:
            image = Image.open(path)
            image = preprocess(image)
            imgs.append(image)

        samples = len(imgs)
        imgs = torch.stack(imgs)

        gts = np.zeros((samples, 1))
        data = np.genfromtxt(csv_path, delimiter=',')

        for i, x in enumerate(data):
            if i < samples:
                gts[i] = x
        gts = gts

        self.x = imgs
        self.y = torch.from_numpy(gts)
        self.n_samples = samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples