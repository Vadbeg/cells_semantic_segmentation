"""Script for instance segmentation using classical CV """


from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from cv2 import cv2

from cells_semantic_segmentation.constants import DeviceType
from cells_semantic_segmentation.modules.evaluation.instance_seg_evaluator import (
    InstanceSegEvaluator,
)
from cells_semantic_segmentation.modules.evaluation.seg_model_evaluator import (
    SegModelEvaluator,
)


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


def _combine_image_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.stack(
        [
            mask * np.random.randint(0, 255),
            mask * np.random.randint(0, 255),
            mask * np.random.randint(0, 255),
        ],
        axis=-1,
    )

    combined_image = np.array(np.uint8(image * 0.6 + mask * 0.4))

    return combined_image


if __name__ == '__main__':
    checkpoint_path = Path(
        '/home/vadbeg/Projects/kaggle/cell_instan'
        'ce_segmentation/cells_semantic_segmentation'
        '/logs/cells-sem-seg/14xtq99r/checkpoints/epoch=30-step=960.ckpt'
    )
    image_path = Path(
        '/home/vadbeg/Data_SSD/Kaggle/cel'
        'l_instance_segmentation/sartoriu'
        's-cell-instance-segmentation/tes'
        't/7ae19de7bc2a.png'
    )

    image_size = (512, 512)

    device = DeviceType('cpu')

    model = _load_model(checkpoint_path=checkpoint_path, device=device)
    image = _load_image(image_path=image_path)

    seg_model_evaluator = SegModelEvaluator(
        model=model, image_size=image_size, device=torch.device(device.value)
    )

    instance_seg_evaluator = InstanceSegEvaluator(
        seg_model_evaluator=seg_model_evaluator
    )

    contours = instance_seg_evaluator.eval(image=image)

    from cv2 import cv2

    res_image = image.copy()
    whole_image_area = image.shape[0] * image.shape[1]

    for curr_contour in contours:
        color = np.random.randint(10, 255, size=3).tolist()

        contour_area = cv2.contourArea(contour=curr_contour)

        if contour_area > whole_image_area / 10 ** 5:
            res_image = cv2.drawContours(
                res_image,
                contours=[curr_contour],
                contourIdx=-1,
                color=color,
                thickness=1,
            )

    plt.subplots(figsize=(12, 12))
    plt.imshow(res_image)
    plt.show()
