"""Module with instance segmentation evaluation based on classical CV"""

from typing import List

import numpy as np
from cv2 import cv2

from cells_semantic_segmentation.modules.evaluation.seg_model_evaluator import (
    SegModelEvaluator,
)


class InstanceSegEvaluator:
    def __init__(self, seg_model_evaluator: SegModelEvaluator):
        self.seg_model_evaluator = seg_model_evaluator

    def eval(self, image: np.ndarray) -> List[np.ndarray]:
        mask = self.seg_model_evaluator.eval(image=image)

        mask = self._erode_mask(mask=mask)
        contours = self._find_contours(mask=mask)

        return contours

    @staticmethod
    def _erode_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.erode(mask[0], kernel=kernel)

        return mask

    @staticmethod
    def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return list(contours)
