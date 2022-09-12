import cv2
import pandas as pd
import numpy as np

import os
import json

import tqdm

def csv_to_img(path):
    data = pd.read_csv(path)
    print(data.shape)
    img = data.to_numpy()
    #norm = np.linalg.norm(img)
    img = img / img.max()
    img = img * 255
    img = img.astype(np.uint8)
    return img


def img_to_num(img):
    th, threshold = cv2.threshold(img, 170, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    print("nums: {}".format(len(cnts)))

    #cv2.imshow("image", img)
    #cv2.waitKey(0)

    # cv2.drawContours(np.vstack((img, np.zeros(img.size), np.zeros(img.size))), np.array(cnts), -1, (0, 255, 0), 1,
    #                  cv2.LINE_AA)
    # cv2.imwrite("dst.png", img)

    return len(cnts)


def convert_all(dir):
    result = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            img = csv_to_img(path)
            num = img_to_num(img)
            result[file] = num
    return result


def write_json(numbers, path):
    with open(os.path.join(path, "count.json"), "w", encoding="utf-8") as file:
        json.dump(numbers, file)
