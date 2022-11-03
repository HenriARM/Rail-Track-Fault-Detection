from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig("transformations.svg")


plt.rcParams["savefig.bbox"] = "tight"
orig_img = Image.open("train/Defective/E116_8996 (1).jpg")
torch.manual_seed(0)
padded_imgs = [
    T.RandomVerticalFlip()(orig_img),
    T.ColorJitter(brightness=0.5, hue=0.3)(orig_img),
    T.RandomRotation(degrees=(0, 40))(orig_img),
    T.RandomAffine(degrees=0, translate=(0.2, 0.2))(orig_img),
    T.RandomAffine(degrees=0, shear=(0.2, 0.2))(orig_img)
]
plot(padded_imgs)
