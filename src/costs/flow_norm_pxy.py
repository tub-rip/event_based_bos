import logging
from typing import Union

import numpy as np
import torch

from . import CostBase, FlowNorm

logger = logging.getLogger(__name__)


class FlowNormPxy(FlowNorm):
    """Norm of the flow
    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "flow_norm_pxy"
    required_keys = ["pxy"]

    def __init__(self, direction="minimize", store_history: bool = False, *args, **kwargs):
        super().__init__(direction=direction, store_history=store_history)

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> Union[float, torch.Tensor]:
        """Calculate norm of the pxy.
        Inputs:
            prediction (np.ndarray or torch.Tensor) ... Should be the same shape as measurement
            measurement (np.ndarray or torch.Tensor) ... Should be the same shap as prediction

        Returns:
            L2 (Union[float, torch.Tensor])
        """
        pxy = arg["pxy"]
        if isinstance(pxy, torch.Tensor):
            return self.calculate_torch(pxy)
        elif isinstance(pxy, np.ndarray):
            return self.calculate_numpy(pxy)
        e = f"Unsupported input type. {type(pxy)}."
        logger.error(e)
        raise NotImplementedError(e)
