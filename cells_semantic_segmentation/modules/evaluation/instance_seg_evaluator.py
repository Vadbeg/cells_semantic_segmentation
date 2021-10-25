"""Module with instance segmentation evaluation based on classical CV"""

from typing import List, Tuple

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
        mask = self._erode_mask(mask=mask[0])

        mask_shape = (mask.shape[0], mask.shape[1])

        contours = self._find_contours(mask=mask)
        res_masks = self._post_process_contours(
            contours=contours, mask_shape=mask_shape
        )

        return res_masks

    @staticmethod
    def _erode_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((6, 6), np.uint8)
        mask = cv2.erode(mask, kernel=kernel)

        return mask

    @staticmethod
    def _find_contours(mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return list(contours)

    @staticmethod
    def _post_process_contours(
        contours: List[np.ndarray],
        mask_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        whole_image_area = mask_shape[0] * mask_shape[1]
        res_masks = []

        for curr_contour in contours:
            contour_area = cv2.contourArea(contour=curr_contour)

            if contour_area > whole_image_area / 10 ** 5:
                temp_mask = np.zeros(shape=mask_shape)
                temp_mask = cv2.drawContours(
                    temp_mask,
                    contours=[curr_contour],
                    contourIdx=-1,
                    color=1,
                    thickness=-1,
                )

                res_masks.append(temp_mask)

        return res_masks
