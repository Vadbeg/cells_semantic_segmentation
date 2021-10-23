"""Module for starting train script"""


import warnings

from pytorch_lightning.utilities.cli import LightningCLI

from cells_semantic_segmentation.modules.train.training import CellsSemSegModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    cli = LightningCLI(model_class=CellsSemSegModel, save_config_callback=None)
