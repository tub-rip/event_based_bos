import logging
from typing import Union

import cv2
import numpy as np
import torch

from ..utils import SobelTorch
from ..visualizer import Visualizer
from . import CostBase

logger = logging.getLogger(__name__)


class ImageGradient(CostBase):
    """Image gradient loss
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "image_gradient"
    required_keys = ["flow", "omit_boundary"]

    def __init__(
        self,
        direction="minimize",
        store_history: bool = False,
        cuda_available=False,
        precision="32",
        visualize_intermediate=False,
        *args,
        **kwargs,
    ):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate gradient of flow
        Inputs:
            flow (np.ndarray or torch.Tensor) ... [(b,) 2, W, H]. Flow of the image.
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            contrast (Union[float, torch.Tensor]) ... contrast of the image.
        """
        flow = arg["flow"]
        omit_boundary = arg["omit_boundary"]
        weights = arg["weights"] # if "weights" in arg.keys() else None

        if isinstance(flow, torch.Tensor):
            return self.calculate_torch(flow, weights, omit_boundary)
        elif isinstance(flow, np.ndarray):
            return self.calculate_numpy(flow, weights, omit_boundary)
        e = f"Unsupported input type. {type(flow)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, flow: torch.Tensor, weights, omit_boundary: bool) -> torch.Tensor:
        """Calculate contrast of the count image.
        Inputs:
            flow (torch.Tensor) ... [2, W, H]. Image of warped events

        Returns:
            loss (torch.Tensor) ... Gradient.
        """
        gradx = torch.gradient(flow, dim=1)[0] * weights
        grady = torch.gradient(flow, dim=2)[0] * weights
        loss = torch.mean(torch.abs(gradx) + torch.abs(grady))

        if self.direction == "minimize":
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss

