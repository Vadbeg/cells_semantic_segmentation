"""Module with cells dataset"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from cv2 import cv2
from torch.utils.data import Dataset

from cells_semantic_segmentation.modules.data.helpers import rle_decode


class CellDataset(Dataset):
    def __init__(
        self,
        images_root_folder: Union[str, Path],
        dataframe: pd.DataFrame,
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Callable] = None,
    ):
        self.image_root_folder = Path(images_root_folder)
        self.dataframe = dataframe
        self.transforms = transforms
        self.image_size = image_size

        self.unique_image_ids = self._get_unique_image_ids(dataframe=dataframe)

    def __getitem__(self, index):
        image_id = self.unique_image_ids[index]

        image = self._load_image(image_id=image_id)
        mask = self._get_mask_from_image_id(image_id=image_id)

        image = self._resize(image=image, new_size=self.image_size)
        mask = self._resize(image=mask, new_size=self.image_size)

        item = {'image': image, 'mask': mask}

        if self.transforms:
            item = self.transforms(**item)

        return item

    def _load_image(self, image_id: str) -> np.ndarray:
        image_file_name = image_id + '.png'
        image_path = self.image_root_folder.joinpath(image_file_name)

        image = cv2.imread(filename=str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _get_mask_from_image_id(self, image_id: str) -> np.ndarray:
        rle_annotations, image_shape = self._get_image_annotations_and_shape(
            image_id=image_id
        )
        mask = self._get_mask_from_rle_annotations(
            rle_annotations=rle_annotations, shape=image_shape
        )

        return mask

    @staticmethod
    def _get_mask_from_rle_annotations(
        rle_annotations: List[str], shape: Tuple[int, int]
    ) -> np.ndarray:
        masks = [
            rle_decode(mask_rle=curr_rle_annotation, shape=shape)
            for curr_rle_annotation in rle_annotations
        ]

        res_mask = np.sum(masks, axis=0)
        res_mask = res_mask.clip(min=0, max=1)

        return res_mask

    def _get_image_annotations_and_shape(
        self, image_id: str
    ) -> Tuple[List[str], Tuple[int, int]]:
        dataframe_part = self.dataframe.loc[self.dataframe['id'] == image_id]
        rle_annotations = dataframe_part['annotation'].tolist()

        shape = (dataframe_part.iloc[0]['height'], dataframe_part.iloc[0]['width'])

        return rle_annotations, shape

    @staticmethod
    def _get_unique_image_ids(dataframe: pd.DataFrame) -> List[str]:
        unique_image_ids = dataframe['id'].unique().tolist()

        return unique_image_ids

    @staticmethod
    def _resize(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        image = cv2.resize(image, new_size)

        return image

    def __len__(self) -> int:
        return len(self.unique_image_ids)
