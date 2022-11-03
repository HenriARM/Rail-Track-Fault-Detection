import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch
import numpy as np

plt.figure(figsize=(15, 12))

transforms = T.Compose([T.ToTensor()])
data_dir = "./test"
data = torchvision.datasets.ImageFolder(data_dir, transform=transforms)
data_loader = torch.utils.data.DataLoader(data)

for idx, t in enumerate(data_loader):
    x, y = t
    if idx == 31:
        break
    print(x.shape)
    plt.subplot(4,8,idx+1)
    image = (x[0] * 255.).data.numpy().astype(np.uint8)
    image = np.moveaxis(image, 0, -1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(idx)
plt.tight_layout()
plt.savefig("test_images.svg")
