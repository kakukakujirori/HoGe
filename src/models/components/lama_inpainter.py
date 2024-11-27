import os
import sys
from urllib.parse import urlparse

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url):
    if os.path.exists(url):
        cached_file = url
    else:
        cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)

    return cached_file


class LamaInpainter(nn.Module):
    def __init__(self):
        super().__init__()
        if os.path.exists(LAMA_MODEL_URL):
            model_path = LAMA_MODEL_URL
        else:
            model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location="cpu")

    def forward(self, img: Float[torch.Tensor, "b 3 h w"], mask: Float[torch.Tensor, "b 1 h w"]):
        return self.model(img, mask)
