import glob
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from jaxtyping import Float
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        max_size: int = 1024,
        augmentation_list: list = [],
    ) -> None:
        self.max_size = max_size
        self.aug = A.Compose(augmentation_list)
        self.image_paths = []
        self.image_paths += glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_paths += glob.glob(os.path.join(image_dir, "*.png"))
        self.image_paths.sort()

    def __getitem__(self, index: int) -> Float[torch.Tensor, "c h w"]:
        # load an image
        img_p = self.image_paths[index]
        img = Image.open(img_p).convert("RGB")
        if max(img.size) > self.max_size:
            width, height = img.size
            if width > height:
                img = img.resize((self.max_size, height * self.max_size // width))
            else:
                img = img.resize((width * self.max_size // height, height))

        # augmentations
        img = np.array(img)
        img = self.aug(image=img)["image"]

        # convert to tensors (NOTE: CHANNEL LAST!!!)
        img_t = torch.from_numpy(img).float() / 255

        return img_t

    def __len__(self):
        return len(self.image_paths)
