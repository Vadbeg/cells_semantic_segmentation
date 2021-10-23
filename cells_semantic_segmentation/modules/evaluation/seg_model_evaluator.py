"""Module with model evaluation"""

from typing import Tuple

import numpy as np
import torch
from cv2 import cv2

from cells_semantic_segmentation.modules.data.utils import get_val_transforms


class SegModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        image_size: Tuple[int, int] = (512, 512),
        edge: float = 0.4,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.image_size = image_size
        self.edge = edge
        self.device = device

        self.transforms = get_val_transforms()

    def eval(self, image: np.ndarray) -> np.ndarray:
        original_image_size = (image.shape[1], image.shape[0])
        image = self._resize(image=image, new_size=self.image_size)

        image_tensor = self.transforms(image=image)['image']
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        mask_tensor: torch.Tensor = self.model(image_tensor)

        mask = mask_tensor.detach().cpu().numpy()
        mask = np.uint8(mask > self.edge)[0, 0, ...]

        mask = self._resize(image=mask, new_size=original_image_size)
        mask = mask[np.newaxis, ...]

        return mask

    @staticmethod
    def _resize(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        image = cv2.resize(image, new_size)

        return image
