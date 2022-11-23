import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import numpy as np


class CustomHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_one_image(self, req):
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, x):
        outs = self.model.to("cpu").forward(x.to("cpu"))
        y_prim = outs
        np_y_prim = torch.sigmoid(y_prim).cpu().data.numpy().flatten()
        np_y_prim = np.rint(np_y_prim).astype(np.uint8)
        return np_y_prim

    def postprocess(self, preds):
        res = [{"class": int(preds[0])}]
        return res
