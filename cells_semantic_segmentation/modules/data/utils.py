"""Module with utils for dataset creation and usage"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def create_data_loader(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader


def get_train_val_dataframes(
    dataframe: pd.DataFrame,
    train_split_percent: float = 0.7,
    shuffle: bool = True,
    item_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_image_ids = dataframe['id'].unique().tolist()
    if shuffle:
        np.random.shuffle(unique_image_ids)

    edge_value = int(train_split_percent * len(unique_image_ids))

    train_image_id = unique_image_ids[:edge_value]
    val_image_id = unique_image_ids[edge_value:]

    if item_limit:
        train_image_id = train_image_id[:item_limit]
        val_image_id = val_image_id[:item_limit]

    train_dataframe = dataframe.loc[dataframe['id'].isin(train_image_id)]
    val_dataframe = dataframe.loc[dataframe['id'].isin(val_image_id)]

    return train_dataframe, val_dataframe
