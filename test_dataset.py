from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cells_semantic_segmentation.modules.data.dataset.cells_dataset import CellDataset
from cells_semantic_segmentation.modules.data.utils import get_train_val_dataframes


def combine_image_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    combined_image = np.array(np.uint8(image * 0.6 + mask * 0.4))

    return combined_image


if __name__ == '__main__':
    data_root: Path = Path(
        '/home/vadbeg/Data_SSD/Kaggle/cell_in'
        'stance_segmentation/sartorius-cell-instance-segmentation'
    )

    train_images_folder = data_root.joinpath('train')
    train_dataframe_path = data_root.joinpath('train.csv')

    dataframe = pd.read_csv(train_dataframe_path)

    train_dataframe, val_dataframe = get_train_val_dataframes(
        dataframe=dataframe, train_split_percent=0.7
    )

    cell_dataset = CellDataset(
        images_root_folder=train_images_folder,
        dataframe=train_dataframe,
        image_size=(512, 512),
    )

    _image = cell_dataset[0]['image']
    _mask = cell_dataset[0]['mask']

    _mask = np.stack(
        [
            _mask * np.random.randint(0, 255),
            _mask * np.random.randint(0, 255),
            _mask * np.random.randint(0, 255),
        ],
        axis=-1,
    )

    combined_image = combine_image_and_mask(image=_image, mask=_mask)

    plt.subplots(figsize=(12, 8))
    plt.imshow(combined_image)
    plt.show()
