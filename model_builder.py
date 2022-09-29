import torch
import torchvision


def build_vis_model(model_name, density_map):
    if density_map:
        pass  # TODO
    else:
        if model_name == "Resnet_18":  # ["Resnet_18", "Resnet_50", "YOLO5S", "FCNresnet50"]
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(512, 1)
        elif model_name == "Resnet_50":
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, 1)
        elif model_name == "YOLO5S":
            pass
        elif model_name == "FCNresnet50":
            pass
        else:
            raise Exception("Unexpected model")

    return model
