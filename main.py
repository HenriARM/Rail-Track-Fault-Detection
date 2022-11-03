import torch
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torchsummary import summary


def main():
    train_dir = "./train"
    test_dir = "./valid"

    # transformations (Torch EfficientNet will automatically do [0.0-1.0] rescaling and normalization), check
    # https://pytorch.org/vision/0.14/models/generated/torchvision.models.efficientnet_b4.html#torchvision.models.efficientnet_b4
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225],
                                                 ),
                                                 ])
    # datasets
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transforms)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # accepts input (B, C, H, W) or (C, H, W)
    model = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1", progress=True)
    summary(model, (3, 224, 224))
    # freeze all params
    for params in model.parameters():
        params.requires_grad_ = False
    NUM_CLASSES = 1
    model.classifier = nn.Sequential(model.classifier[0],
                                     nn.Linear(in_features=model.classifier[-1].in_features,
                                               out_features=NUM_CLASSES, bias=True))
    model = model.to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())

    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    epochs = 10
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03

    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            x = x.to(device)
            y = y.unsqueeze(1).float()  # convert target to same nn output shape
            y = y.to(device)  # move to gpu

            # make prediction
            yhat = model(x)
            # enter train mode
            model.train()
            # compute loss
            loss = loss_fn(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # optimizer.cleargrads()
            # TODO: / batch
            epoch_loss += loss / len(train_loader)
            losses.append(loss)

        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch + 1, epoch_loss))

        # validation doesnt requires gradient
        with torch.no_grad():
            cum_loss = 0
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
                y_batch = y_batch.to(device)

                # model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += loss / len(test_loader)
                val_losses.append(val_loss.item())

            epoch_test_losses.append(cum_loss)
            print('Epoch : {}, val loss : {}'.format(epoch + 1, cum_loss))

            best_loss = min(epoch_test_losses)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()

            # early stopping
            early_stopping_counter = 0
            if cum_loss > best_loss:
                early_stopping_counter += 1

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("/nTerminating: early stopping")
                break  # terminate training

    # load best model
    model.load_state_dict(best_model_wts)


if __name__ == '__main__':
    main()

# TODO: plot train, valid loss
# TODO: separate eval.py which loads model, runs inference,
#  prints statistics (confusion matrix and classification report)