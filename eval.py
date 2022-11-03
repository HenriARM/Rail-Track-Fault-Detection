import numpy as np
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torchvision.transforms as T
import os

device = "cpu"
model_filename = "./exp_eff_b7_lr_e3/best_model.pt"
model = torch.load(model_filename, map_location="cpu")
model = model.eval()
model = model.to(device)

transforms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
data_dir = "./test"
data = torchvision.datasets.ImageFolder(data_dir, transform=transforms)
data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=22)

for x,y in data_loader:
    x = x.to(device)
    y = y.unsqueeze(1).float()
    y = y.to(device)

    y_prim = model.forward(x)
    np_y_prim = torch.sigmoid(y_prim).cpu().data.numpy().flatten()
    np_y_prim = np.rint(np_y_prim).astype(np.uint8)
    np_y = y.cpu().data.numpy().flatten().astype(np.uint8)
    print(f"Y: {np_y}")
    print(f"Y prim: {np_y_prim}")

    labels = ["Defective", "Non Defective"]
    cm = confusion_matrix(np_y, np_y_prim)
    print("Confusion matrix", cm)
    print(classification_report(np_y, np_y_prim, target_names=labels))
