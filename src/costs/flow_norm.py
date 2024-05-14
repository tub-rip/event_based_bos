import logging
from typing import Union

import numpy as np
import torch

from . import CostBase

logger = logging.getLogger(__name__)


class FlowNorm(CostBase):
    """Norm of the flow
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "flow_norm"
    required_keys = ["flow"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate norm of the flow.
        Inputs:
            prediction (np.ndarray or torch.Tensor) ... Should be the same shape as measurement
            measurement (np.ndarray or torch.Tensor) ... Should be the same shap as prediction

        Returns:
            L1 (Union[float, torch.Tensor])
        """
        flow = arg["flow"]
        if isinstance(flow, torch.Tensor):
            return self.calculate_torch(flow)
        elif isinstance(flow, np.ndarray):
            return self.calculate_numpy(flow)
        e = f"Unsupported input type. {type(flow)}."
        logger.error(e)
        raise NotImplementedError(e)

    def calculate_torch(self, flow: torch.Tensor) -> torch.Tensor:
        """Calculate L2 norm.
        """
        # if weights is None:
        #     weights = torch.ones_like(prediction)
        # print('-=-=--=', flow.shape, flow.max(), flow.min())
        # raise RuntimeError
        loss = torch.linalg.norm(flow, dim=0).mean()
        if self.direction == "minimize":
            return loss
        logger.warning("The loss is specified as maximize direction")
        return -loss

    def calculate_numpy(self, flow: np.ndarray) -> float:
        """Calculate L2 norm.
        """
        loss = np.linalg.norm(flow, axis=0).mean()
        if self.direction == "minimize":
            return loss
        return loss
