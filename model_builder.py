import torch
import torchvision

from models.models import *
from models.modules import CrowdCountHeader, DensityMapHeader


def build_vis_model(model_name, density_map):
    header = DensityMapHeader if density_map else CrowdCountHeader

    if model_name == "Resnet_18":  # ["Resnet_18", "Resnet_50", "YOLO5S", "FCNresnet50"]
        par = None if density_map else {"in_features": 512}
        model = ResNet18(header, **par)
    elif model_name == "Resnet_50":
        par = None if density_map else {"in_features": 2048}
        model = ResNet50(header, **par)
    elif model_name == "YOLO5S":
        par = None if density_map else {"in_features": 512}
        model = YOLOv5S(header, **par)
    elif model_name == "Unet":
        par = None if density_map else {"in_features": 64}
        model = UNet(header, **par)
    else:
        raise Exception("Unexpected model")

    return model
