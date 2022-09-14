import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import copy

from data_loader import MallDataSet


def train(model, data_loader, optimizer, criterion, device, epochs=100, train_loader=None):

    loader = {
        "train": data_loader,
        "test": train_loader,
    }
    loss_hist = []
    least_loss = 999999999
    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 20)

        for phase in ["train", "test"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for images, gts in loader[phase]:
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
            epoch_loss = running_loss / len(loader[phase].dataset)
            print(f"{phase} loss: {epoch_loss}")

            if phase == "test":
                loss_hist.append(epoch_loss)
                if least_loss > epoch_loss:
                    least_loss = epoch_loss
                    best_model = copy.deepcopy(model.state_dict())

    return best_model


if __name__ == "__main__":

    parameters_path = "resnet18_reg1.pt"

    data = MallDataSet()
    l = len(data)
    train_size = int(0.8 * l)
    train_d, test_d = torch.utils.data.random_split(data, [train_size, l - train_size])

    train_loader = DataLoader(dataset=train_d, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_d, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    model_dict = train(model, train_loader, optimizer, criterion, device, train_loader=train_loader)

    torch.save(model_dict, parameters_path)
