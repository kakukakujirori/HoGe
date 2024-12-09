import glob
import os

import albumentations as A
import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        height: int = 512,
        width: int = 512,
        include_alpha_channel: bool = False,
        augmentation_list: list = [],
    ) -> None:
        self.height = height
        self.width = width
        self.include_alpha_channel = include_alpha_channel
        self.aug = A.Compose(augmentation_list)
        self.image_paths = []
        self.image_paths += glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_paths += glob.glob(os.path.join(image_dir, "*.png"))
        self.image_paths.sort()

    def __getitem__(self, index: int) -> Float[torch.Tensor, "c h w"]:
        # load an image
        img_p = self.image_paths[index]
        if self.include_alpha_channel:
            img = Image.open(img_p).convert("RGBA")
        else:
            img = Image.open(img_p).convert("RGB")
        img = img.resize((self.width, self.height))

        # augmentations
        img = np.array(img)
        img = self.aug(image=img)["image"]

        # convert to tensors (NOTE: CHANNEL LAST!!!)
        img_t = torch.from_numpy(img).float() / 255

        return img_t

    def __len__(self):
        return len(self.image_paths)
