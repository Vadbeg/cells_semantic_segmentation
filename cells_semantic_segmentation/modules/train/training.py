"""Module for semantic segmentation model training"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from cells_semantic_segmentation.modules.data.dataset.cells_dataset import CellDataset
from cells_semantic_segmentation.modules.data.utils import (
    create_data_loader,
    get_train_transforms,
    get_train_val_dataframes,
    get_val_transforms,
)


class CellsSemSegModel(pl.LightningModule):
    def __init__(
        self,
        images_root_folder: str,
        dataframe_path: str,
        shuffle_dataset: bool = True,
        train_split_percent: float = 0.7,
        dataset_items_limit: Optional[int] = None,
        image_size: Tuple[int, int] = (512, 512),
        backbone: str = 'resnet18',
        in_channels: int = 3,
        classes: int = 1,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        learning_rate: float = 0.0001,
    ):
        super().__init__()

        self.images_root_folder = images_root_folder
        self.image_size = image_size

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate

        self.train_dataframe, self.val_dataframe = self._load_train_val_dataframes(
            dataframe_path=Path(dataframe_path),
            train_split_percent=train_split_percent,
            shuffle=shuffle_dataset,
            item_limit=dataset_items_limit,
        )

        self.loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.model = smp.Unet(backbone, in_channels=in_channels, classes=classes)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_id: int
    ) -> torch.Tensor:  # pylint: disable=W0613
        image = batch['image']
        mask = batch['mask']

        prediction = self.model(image)
        loss_value = self.loss(y_pred=prediction, y_true=mask)

        self.log(
            name='train_loss',
            value=loss_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return loss_value

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_id: int
    ) -> torch.Tensor:  # pylint: disable=W0613
        image = batch['image']
        mask = batch['mask']

        prediction = self.model(image)
        loss_value = self.loss(y_pred=prediction, y_true=mask)

        self.log(
            name='val_loss', value=loss_value, prog_bar=True, logger=True, on_epoch=True
        )

        return loss_value

    def train_dataloader(self) -> DataLoader:
        train_transforms = get_train_transforms()

        train_dataset = CellDataset(
            images_root_folder=self.images_root_folder,
            dataframe=self.train_dataframe,
            image_size=self.image_size,
            transforms=train_transforms,
        )
        train_dataloader = create_data_loader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_transforms = get_val_transforms()

        val_dataset = CellDataset(
            images_root_folder=self.images_root_folder,
            dataframe=self.val_dataframe,
            image_size=self.image_size,
            transforms=val_transforms,
        )
        val_dataloader = create_data_loader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=2,
        )

        return val_dataloader

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, mode='min'
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    @staticmethod
    def _load_train_val_dataframes(
        dataframe_path: Path,
        train_split_percent: float = 0.7,
        shuffle: bool = True,
        item_limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not dataframe_path.exists():
            raise ValueError(f'Bad csv path: {dataframe_path}')

        dataframe = pd.read_csv(dataframe_path)
        train_dataframe, val_dataframe = get_train_val_dataframes(
            dataframe=dataframe,
            train_split_percent=train_split_percent,
            shuffle=shuffle,
            item_limit=item_limit,
        )

        return train_dataframe, val_dataframe
