import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
# from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib

import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    train_dir = "./train"
    test_dir = "./valid"

    # transformations (Torch EfficientNet will automatically do [0.0-1.0] rescaling and normalization), check
    # https://pytorch.org/vision/0.14/models/generated/torchvision.models.efficientnet_b4.html#torchvision.models.efficientnet_b4
    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=.5, hue=.3),
            # T.RandomPerspective(),
            T.RandomRotation(degrees=(0, 40)),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2)), # shift
            T.RandomAffine(degrees=0, shear=(0.2, 0.2)), # shear
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transforms = T.Compose(
        [
            T.RandomVerticalFlip(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
                                                 
    # datasets
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # accepts input (B, C, H, W) or (C, H, W)
    model = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1", progress=True)
    # freeze all params
    for params in model.parameters():
        params.requires_grad_ = False
    NUM_CLASSES = 1
    model.classifier = nn.Sequential(model.classifier[0],
                                     nn.Linear(in_features=model.classifier[-1].in_features,
                                               out_features=NUM_CLASSES, bias=True))
    model = model.to(device)
    summary(model, (3, 224, 224))
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())

    metrics = {}
    for stage in ["train", "test"]:
        for metric in ["loss", "acc"]:
            metrics[f"{stage}_{metric}"] = []

    epochs = 100
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03

    for epoch in range(epochs):
        plt.clf()
        for loader in [train_loader, test_loader]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            if loader == train_loader:
                stage = "train"
                model = model.train()
                torch.set_grad_enabled(True)
            else:
                stage = "test"
                model = model.eval()
                torch.set_grad_enabled(False)

            for x, y in loader:
                x = x.to(device)
                y = y.unsqueeze(1).float()
                y = y.to(device)

                y_prim = model.forward(x)
                loss = loss_fn(y_prim, y)
                metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

                if loader == train_loader:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                np_y_prim = torch.sigmoid(y_prim).cpu().data.numpy()
                np_y = y.cpu().data.numpy()
                acc = np.mean((np_y == np.rint(np_y_prim)) * 1.0)
                metrics_epoch[f'{stage}_acc'].append(acc)

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')
        print("\n")
        best_loss = min(metrics["test_loss"])

        # save best model
        if metrics["test_loss"][-1] <= best_loss:
            best_model_wts = model  # .state_dict()

        # early stopping
        early_stopping_counter = 0
        if metrics["test_loss"][-1] > best_loss:
            early_stopping_counter += 1

        if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            print("/nTerminating: early stopping")
            break  # terminate training

        plt.clf()
        plts = []
        plt.subplot(2, 1, 1)
        plt.title("loss")
        plts += plt.plot(metrics["train_loss"], label="train_loss")
        plts += plt.plot(metrics["test_loss"], label="test_loss")
        plt.legend(plts, [it.get_label() for it in plts])

        plts = []
        plt.subplot(2, 1, 2)
        plt.title("acc")
        plts += plt.plot(metrics["train_acc"], label="train_acc")
        plts += plt.plot(metrics["test_acc"], label="test_acc")
        plt.legend(plts, [it.get_label() for it in plts])
        plt.savefig("train.svg")

    # save best model
    # (two options how to do that https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch)
    torch.save(best_model_wts, "./best_model.pt")


if __name__ == '__main__':
    main()

# TODO: create eval.py which loads model, runs inference,
#  prints statistics (confusion matrix and classification report)
