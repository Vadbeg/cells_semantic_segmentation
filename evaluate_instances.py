"""Script to start simple sem seg model evaluation"""

import typer

from cells_semantic_segmentation.cli.perform_instance_seg import (
    start_instance_segmentation,
)

if __name__ == '__main__':
    typer.run(start_instance_segmentation)
