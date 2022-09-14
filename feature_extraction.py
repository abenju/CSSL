import numpy as np
import os
import argparse
from PIL import Image


import torch
import torchvision
from torchvision import transforms

ACCEPTED_MODELS = [
    "YOLOV5N",
    "RESNET18"
]

ROOT_SUBDIR = "\\input"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--DATA_ROOT', help='Location to root directory for dataset reading')
parser.add_argument('--SAVE_ROOT', help='Location to root directory for saving checkpoint models')
parser.add_argument('--MODEL', help='Model name')

args = parser.parse_args()

model_name = args.MODEL

if model_name == "YOLOV5N":
    pass
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    # print(*list(model.children()))
elif model_name == "RESNET18":
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 1)

    transform = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pt_path = "resnet18_reg1.pt"
    model.load_state_dict(torch.load(pt_path))
    model = torch.nn.Sequential(*list(model.children())[:-1])

else:
    raise Exception("Model not supported")

model = model.to(DEVICE)


if not os.path.exists(args.SAVE_ROOT+'/'+'Feature_files'):
    os.makedirs(args.SAVE_ROOT+'/'+'Feature_files')

for subdir, dirs, files in os.walk(args.DATA_ROOT + ROOT_SUBDIR):
    for file in files:
        print(file)
        f = open(args.SAVE_ROOT + '\\' + 'Feature_files\\' + file + '.txt', "w")
        img_path = os.path.join(args.DATA_ROOT + ROOT_SUBDIR, file)
        img_name = os.path.basename(img_path)
        image = Image.open(img_path)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.cuda()
        output = model(image)  # output now has the features corresponding to input x
        output = torch.squeeze(output)
        f.write("mall" + ',' + img_name + ',')
        for k in output:
            f.write(str(np.float(k)) + ',')
        f.write('\n')
        f.close()
