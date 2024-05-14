import copy
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import scipy.signal


@dataclass
class FlowPatch:
    """Dataclass for flow patch"""

    # center of coordinates
    x: np.int16  # height
    y: np.int16  # width
    shape: tuple  # (height, width)
    # flow (pixel displacement) value of the flow at the location
    u: float = 0.0  # height
    v: float = 0.0  # width

    # To be compatible with dictionary expression..
    def __getitem__(self, item):
         return getattr(self, item)

    @property
    def h(self) -> int:
        return self.shape[0]

    @property
    def w(self) -> int:
        return self.shape[1]

    @property
    def x_min(self) -> int:
        return int(self.x - np.ceil(self.h / 2))

    @property
    def x_max(self) -> int:
        return int(self.x + np.floor(self.h / 2))

    @property
    def y_min(self) -> int:
        return int(self.y - np.ceil(self.w / 2))

    @property
    def y_max(self) -> int:
        return int(self.y + np.floor(self.w / 2))

    @property
    def xmin(self) -> int:
        return self.x_min

    @property
    def xmax(self) -> int:
        return self.x_max

    @property
    def ymin(self) -> int:
        return self.y_min

    @property
    def ymax(self) -> int:
        return self.y_max

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def flow(self) -> np.ndarray:
        return np.array([self.u, self.v])

    def update_flow(self, u: float, v: float):
        self.u = u
        self.v = v

    def apply_filter(self):
        """Apply filter and get convolved flow.

        Returns:
            [type]: [description]
        """
        flow_u = self._unit_matrix_u * self.u
        flow_v = self._unit_matrix_v * self.v
        return np.concatenate([flow_u[None, ...], flow_v[None, ...]], axis=0)

    def new_ones(self):
        return np.ones(self.shape)

    def copy(self) -> Any:
        return copy.deepcopy(self)
