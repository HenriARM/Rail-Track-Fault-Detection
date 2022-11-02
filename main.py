import torch
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler


def inference(test_data):
    idx = torch.randint(1, len(test_data), (1,))
    sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)
    if torch.sigmoid(model(sample)) < 0.5:
        print("Prediction : Cat")
    else:
        print("Prediction : Dog")
    plt.imshow(test_data[idx][0].permute(1, 2, 0))


def main():
    traindir = "./train"
    testdir = "./test"
    # TODO: add validdir
    # TODO: why dataset is divided into train/test/val? how they are different? what is ratio?
    # transformations
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225],
                                                       ),
                                                       ])
    test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225],
                                                      ),
                                                      ])
    # datasets
    train_data = torchvision.datasets.ImageFolder(traindir, transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(testdir, transform=test_transforms)
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.efficientnet_b4(pretrained=True)
    # freeze all params
    # TODO: check wether others also turn off all layers
    for params in model.parameters():
        params.requires_grad_ = False
    nr_filters = model.fc.in_features  # number of input features of last layer
    # TODO: should I add separate FN layer or edit last one
    model.fc = nn.Linear(nr_filters, 1)
    model = model.to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters())

    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    n_epochs = 10
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03

    for epoch in range(n_epochs):
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            # TODO: custom dataset add .to as transform
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
