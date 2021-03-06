"""CLI for instance segmentation using classical CV"""

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import typer
from cv2 import cv2

from cells_semantic_segmentation.constants import DeviceType
from cells_semantic_segmentation.modules.evaluation.instance_seg_evaluator import (
    InstanceSegEvaluator,
)
from cells_semantic_segmentation.modules.evaluation.seg_model_evaluator import (
    SegModelEvaluator,
)


def start_instance_segmentation(
    checkpoint_path: Path = typer.Option(
        ..., help='Path to pytorch_lightning checkpoint'
    ),
    image_path: Path = typer.Option(..., help='Path to image to segment'),
    image_size: Tuple[int, int] = typer.Option(
        default=(512, 512), help='Size of CT image'
    ),
    device: DeviceType = typer.Option(
        default='cpu', help=f'Device type to use during model execution'
    ),
) -> None:
    _perform_instance_segmentation(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        device=device,
        image_size=image_size,
    )


def _perform_instance_segmentation(
    checkpoint_path: Path,
    image_path: Path,
    device: DeviceType,
    image_size: Tuple[int, int],
) -> None:
    model = _load_model(checkpoint_path=checkpoint_path, device=device)
    image = _load_image(image_path=image_path)

    seg_model_evaluator = SegModelEvaluator(
        model=model, image_size=image_size, device=torch.device(device.value)
    )
    instance_seg_evaluator = InstanceSegEvaluator(
        seg_model_evaluator=seg_model_evaluator
    )

    masks = instance_seg_evaluator.eval(image=image)
    combined_image = _combine_image_and_masks(image=image, masks=masks)

    plt.subplots(figsize=(12, 12))
    plt.imshow(combined_image)
    plt.show()


def _rename_keys(
    state_dict: OrderedDict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    new_state_dict = OrderedDict()

    for layer_name, layer_weights in state_dict.items():
        layer_name = layer_name.replace('model.', '')

        new_state_dict[layer_name] = layer_weights

    return new_state_dict


def _load_model(checkpoint_path: Path, device: DeviceType) -> torch.nn.Module:
    model_meta = torch.load(str(checkpoint_path))
    model_state_dict = model_meta['state_dict']
    model_state_dict = _rename_keys(state_dict=model_state_dict)

    model = smp.Unet('resnet18', in_channels=3, classes=1)
    model.load_state_dict(model_state_dict)
    model.to(device=torch.device(device.value))
    model.eval()

    return model


def _load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(filename=str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _combine_image_and_masks(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    res_mask = np.zeros_like(image)

    for mask in masks[10:]:
        mask = np.stack(
            [
                mask * np.random.randint(0, 255),
                mask * np.random.randint(0, 255),
                mask * np.random.randint(0, 255),
            ],
            axis=-1,
        )
        res_mask += np.uint8(mask)

    res_mask = res_mask.clip(min=0, max=255)
    combined_image = np.array(np.uint8(image * 0.6 + res_mask * 0.4))

    return combined_image
