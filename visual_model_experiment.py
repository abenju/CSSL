import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from HGMM import *
from data_loader import MallDataset
from model_builder import *
import wandb


class AutoDict(dict):
    def __missing__(self, k):
        self[k] = AutoDict()
        return self[k]


def train(model, data_loader, optimizer, criterion, device, epochs=100):
    wandb.watch(model, criterion, log="all", log_freq=10)
    least_loss = 999999999
    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 20)

        train_loss = 0
        val_loss = 0
        test_loss = 0

        for phase in ["train", "val", "test"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for images, gts in data_loader[phase]:
                images = images.to(device)
                gts = gts.to(device, dtype=torch.float32)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    output = model(images)
                    loss = criterion(output, gts)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            print(f"{phase} loss: {epoch_loss}")

            if phase == "val":
                val_loss = epoch_loss
            elif phase == "test":
                test_loss = epoch_loss
                if epoch_loss < least_loss:
                    least_loss = epoch_loss
                    torch.onnx.export(model, images, "model.onnx")
                    torch.save(model.state_dict(), f"saves\\{experiment_name}.pt")
                    #wandb.save(f"{experiment_name}.pt")
            else:
                train_loss = epoch_loss

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss})


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Visual model experiments')
    parser.add_argument("--model", choices=["Resnet_18", "Resnet_50", "YOLO5S", "Unet"], required=True)
    parser.add_argument("--density_map", action="store_true")
    parser.add_argument("--train_on", choices=["mall", "fdst", "ucsd"], nargs='*', required=True)
    parser.add_argument("--test_on", choices=["mall", "fdst", "ucsd"], nargs='*')
    parser.add_argument("--split", type=float, nargs='*', required=True)  # train test val || train val
    #parser.add_argument()

    args = parser.parse_args()

    # Other variables and constants
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = f"{args.model}_{'map' if args.density_map else 'count'}_train_{'_'.join(args.train_on)}"

    lr = 0.001
    epochs = 100
    batch_size = 4

    config = dict(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model=args.model,
        density_map=args.density_map,
        device=device,
        train_on=args.train_on,
        test_on=args.test_on,
        split=args.split,
        momentum=0.9,
        loss="MSE",
        optimizer="SGD",
    )


    # Collect data

    #  TODO: adapt per model
    preprocess = transforms.Compose([
        transforms.Resize((640, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])


    train_d = MallDataset(preprocess, args.train_on, is_map=args.density_map)
    l = len(train_d)
    train_size = int(args.split[0] * l)
    if len(args.split) == 2:
        test_d = MallDataset(preprocess, args.test_on, is_map=args.density_map)
        train_d, val_d = torch.utils.data.random_split(train_d, [train_size, l - train_size])
    elif len(args.split) == 3:
        test_size = int(args.split[1] * l)
        train_d, test_d, val_d = torch.utils.data.random_split(train_d, [train_size, test_size, (l - (train_size + test_size))])
    else:
        raise Exception("Invalid number of splits")

    data = {
        "train": DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, num_workers=2),
        "test": DataLoader(dataset=test_d, batch_size=batch_size, shuffle=True, num_workers=2),
        "val": DataLoader(dataset=val_d, batch_size=batch_size, shuffle=True, num_workers=2),
    }

    # Build Model
    model = build_vis_model(args.model, args.density_map)

    # Train and test

    with wandb.init(project="CSSL", entity="abenju", name=experiment_name, config=config):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

        train(model, data, optimizer, criterion, device, epochs=config["epochs"])
