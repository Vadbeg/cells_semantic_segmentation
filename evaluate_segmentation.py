"""Script to start simple sem seg model evaluation"""

import typer

from cells_semantic_segmentation.cli.perform_segmentation import start_segmentation

if __name__ == '__main__':
    typer.run(start_segmentation)
