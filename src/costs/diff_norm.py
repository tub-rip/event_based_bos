import logging
from typing import Union

import numpy as np
import torch

from . import CostBase

logger = logging.getLogger(__name__)


class DifferenceNorm(CostBase):
    """Norm of prediction and measurement
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "diff_norm"
    required_keys = ["prediction", "measurement"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate L1 loss.
        Inputs:
            prediction (np.ndarray or torch.Tensor) ... Should be the same shape as measurement
            measurement (np.ndarray or torch.Tensor) ... Should be the same shap as prediction

        Returns:
            L1 (Union[float, torch.Tensor])
        """
        prediction = arg["prediction"]
        measurement = arg["measurement"]
        weights = arg["weights"] # if "weights" in arg.keys() else None
        if isinstance(prediction, torch.Tensor):
            return self.calculate_torch(prediction, measurement, weights)
        elif isinstance(prediction, np.ndarray):
            return self.calculate_numpy(prediction, measurement, weights)
        e = f"Unsupported input type. {type(prediction)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, prediction: torch.Tensor, measurement: torch.Tensor, weights) -> torch.Tensor:
        """Calculate L2 norm.
        """
        # if weights is None:
        #     weights = torch.ones_like(prediction)
        loss = torch.linalg.norm(prediction - measurement, ord=1)
        # loss = torch.linalg.norm((prediction - measurement) * weights, ord=None)
        if self.direction == "minimize":
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss

    def calculate_numpy(self, prediction: np.ndarray, measurement: np.ndarray,
                        weights: np.ndarray) -> float:
        """Calculate L2 norm.
        """
        loss = np.linalg.norm(prediction - measurement, ord=1)
        # loss = np.linalg.norm((prediction - measurement) * weights, ord=None)
        if self.direction == "minimize":
            return loss
        return loss
