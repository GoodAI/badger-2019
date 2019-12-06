import abc
from pathlib import Path
from typing import List, Tuple, Optional

import PIL
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset

from attention_sgd.utils.plot_utils import image_to_tensor


class ImageDataset(Dataset):
    images: List

    def __init__(self, images: List, transform):
        self.images = images
        self.transform = transform

    def __getitem__(self, item: int):
        image = self.images[item]
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.images)

    @classmethod
    def from_filenames(cls, root: str, files: List[str], transform=None):
        root_dir = Path(root)
        images = [Image.open((root_dir / file).open('rb')) for file in map(str.strip, files)]
        return cls(images, transform)
