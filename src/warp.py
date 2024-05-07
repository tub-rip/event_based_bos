import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    e = "Torch is disabled."
    logger.warning(e)


from .types import FLOAT_TORCH, NUMPY_TORCH, is_numpy, is_torch, nt_max, nt_min


class MotionModelKeyError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        logger.error(message)
        super().__init__(message)


class FeatureCalculatorMock:
    def __init__(self, *args, **kwargs):
        """Mock class -- please ignore me."""
        pass

    def skip(self):
        feature = {
            "determinant": {"per_event": True, "value": None},
            "trace": {"per_event": True, "value": None},
            "divergence": {"per_event": True, "value": None},
            "straint": {"per_event": True, "value": None},
            "absement": {"per_event": False, "value": None},
        }
        return feature

    def calculate_feature(self, *args, skip: bool = False, **kwargs) -> dict:
        """Mock function

        Returns:
            dict: current supported keys are "determinant", "trace", "divergence".
        """
        if not skip:
            logger.warning(
                "Feature calculation is disabled in this source code!"
            )
        return self.skip()




class Warp(object):
    """Warp functions class.
    It includes various warp function with different motion models.

    Args:
        image_size (tuple[int, int]) ... Image shape. Height, Width. It is used to calculate
            center of the image (cx, cy).
        calculate_feature (bool) ... True to return features related to the warp.
        normalize_t (bool) ...  Defaults to False
    """

    def __init__(
        self,
        image_size: tuple,
        calculate_feature: bool = False,
        normalize_t: bool = False,
        calib_param: Optional[np.ndarray] = None,
    ):
        self.update_property(image_size, calculate_feature, normalize_t, calib_param)
        self.feature_2dof = FeatureCalculatorMock()
        self.feature_dense = FeatureCalculatorMock()

    # Helper functions
    def update_property(
        self,
        image_size: Optional[tuple] = None,
        calculate_feature: Optional[bool] = None,
        normalize_t: Optional[bool] = None,
        calib_param: Optional[np.ndarray] = None,
    ):
        if image_size is not None:
            self.image_size = image_size
        if calculate_feature is not None:
            self.calculate_feature = calculate_feature
        if normalize_t is not None:
            self.normalize_t = normalize_t
        if calib_param is not None:
            logger.info("Set camera matrix K.")
            self.calib_param = calib_param

    def get_key_names(self, motion_model: str) -> list:
        """Returns key name for the motion model.

        Args:
            motion_model (str): "2d-translation" etc.

        Returns:
            list: List of key names.
        """
        if motion_model == "dense-flow":
            e = f"Assume only rigid transformation {motion_model = }"
            logger.warning(e)
            return ["trans_x", "trans_y"]
        elif motion_model in ["2d-translation", "rigid-optical-flow"]:
            return ["trans_x", "trans_y"]
        elif motion_model in ["scaler"]:
            return ["scaler"]
        raise MotionModelKeyError(f"{motion_model = } not supported")

    def get_motion_vector_size(self, motion_model: str) -> int:
        """Returns motion vector size.

        Args:
            motion_model (str): "2d-translation" etc.

        Returns:
            int: Size of the motion vector (DoF).
        """
        params = {k: 0.0 for k in self.get_key_names(motion_model)}
        return len(self.motion_model_to_motion(motion_model, params))

    def motion_model_to_motion(self, motion_model: str, params: dict) -> np.ndarray:
        """Composites motion array from parameter dict.

        Args:
            motion_model (str): "2d-translation" etc.
            params (dict): {param_name: value}

        Returns:
            np.ndarray: Motion vector.
        """
        if motion_model == "dense-flow":
            e = f"Assume only rigid transformation {motion_model = }"
            logger.warning(e)
            motion = np.array([params["trans_x"], params["trans_y"]])
            return self.get_flow_from_motion(motion, "2d-translation")
        elif motion_model in ["2d-translation", "rigid-optical-flow"]:
            return np.array([params["trans_x"], params["trans_y"]])
        elif motion_model in ["scaler"]:
            return np.array([params["scaler"]])
        raise MotionModelKeyError(f"{motion_model = } not supported")

    def motion_model_from_motion(self, motion: np.ndarray, motion_model: str) -> dict:
        """Composites motion model dict from motion. Inverse of `motion_model_to_motion`.

        Args:
            motion (np.ndarray): motion array.
            motion_model (str): "2d-translation" etc.

        Returns:
            (dict): Motion parameter dict.
        """
        if motion_model == "dense-flow":
            e = f"Assume only rigid transformation {motion_model = }"
            logger.warning(e)
            return {"trans_x": motion[0], "trans_y": motion[1]}
        elif motion_model in ["2d-translation", "rigid-optical-flow"]:
            return {"trans_x": motion[0], "trans_y": motion[1]}
        elif motion_model in ["scaler"]:
            return {"scaler": motion[0]}
        raise MotionModelKeyError(f"{motion_model = } not supported")

    def get_flow_from_motion(self, motion: np.ndarray, motion_model: str) -> np.ndarray:
        """Calculate dense flow from motion numerically.

        Args:
            motion (np.ndarray): [description]
            motion_model (str): [description]

        Returns:
            np.ndarray: flow array, 2 x H x W. pix/sec.
        """
        x_range = np.arange(0, self.image_size[0])
        y_range = np.arange(0, self.image_size[1])
        events = np.array([[x, y, 1.0, 1] for x in x_range for y in y_range])
        events = np.concatenate([np.array([[0, 0, 0, 0]]), events])
        if is_torch(motion):
            events = torch.from_numpy(events)
        warped_events, _ = self.warp_event(events, motion, motion_model)
        events = events[1:]
        warped_events = warped_events[1:]
        u = -(warped_events[:, 0] - events[:, 0]).reshape(self.image_size)[None, ...]
        v = -(warped_events[:, 1] - events[:, 1]).reshape(self.image_size)[None, ...]
        if is_torch(motion):
            return torch.cat([u, v], dim=0)
        return np.concatenate([u, v], axis=0)

    # Functions for both numpy and torch arrays
    def warp_event(
        self,
        events: NUMPY_TORCH,
        motion: NUMPY_TORCH,
        motion_model: str,
        direction: Union[str, float] = "first",
        flow_propagate_bin: Optional[int] = None,
    ) -> Tuple[NUMPY_TORCH, dict]:
        """Warp events using optical flow.

        Inputs:
            events (NUMPY_TORCH) ... [(b,) n_events, 4]. Batch of events.
            motion (NUMPY_TORCH) ... [(b,) motion_size ] corresponding to motion_model.
            motion_model (str) ... motion model name. Currently supporting:
                "dense-flow":
                "2d-translation", "rigid-optical-flow":
            direction: Union[str, float] ... For str, 'first', 'middle', 'last', 'random', 'before', 'after' are available.
                For float, it specifies normalized time location.
            flow_propagate_bin (Optional[int]) ... Only effective when motion_model is `dense-flow-voxel-optimized`.

        Returns:
            warped (NUMPY_TORCH) ... [(b,) n_events, 4]. Warped event. (x', y', time, p)
            feature (dict) ... Feature dict.
        """
        # Both numpy and torch coming in here
        ref_time = self.calculate_reftime(events, direction)

        if len(events.shape) == 3:
            ref_time = ref_time[..., None]

        if motion_model == "dense-flow":
            return self.warp_event_from_optical_flow(events, motion, ref_time)
        elif motion_model in ["2d-translation", "rigid-optical-flow"]:
            assert motion.shape[-1] == 2
            return self.warp_event_2dof_xy(events, motion, ref_time)
        raise MotionModelKeyError(f"{motion_model = } not supported")

    def calculate_reftime(
        self, events: NUMPY_TORCH, direction: Union[str, float] = "first"
    ) -> FLOAT_TORCH:
        """Calculate reference time for the warp.

        Args:
            events (NUMPY_TORCH): _description_
            direction (Union[str, float], optional): If float, it calculates the relative direction.
                0 is equivalent to 'first', 0.5 is equivalent to 'middle', and 1.0 is equivalent to 'last'.
                For string inputs, it accepts 'first', 'middle', 'last', 'random', 'before' (-1.0), and 'after' (2.0).
                Defaults to "first".

        Returns:
            NUMPY_TORCH: Reference time scalar, float or torch.float type.
        """
        if type(direction) is float:
            per = nt_max(events[..., 2], -1) - nt_min(events[..., 2], -1)
            return nt_min(events[..., 2], -1) + per * direction
        elif direction == "first":
            return nt_min(events[..., 2], -1)
        elif direction == "middle":
            return self.calculate_reftime(events, 0.5)
        elif direction == "last":
            return nt_max(events[..., 2], -1)
        elif direction == "random":
            return self.calculate_reftime(events, np.random.uniform(low=0.0, high=1.0))
        elif direction == "before":
            return self.calculate_reftime(events, -1.0)
        elif direction == "after":
            return self.calculate_reftime(events, 2.0)
        e = f"direction argument should be first, middle, last. Or float. {direction}"
        logger.error(e)
        raise ValueError(e)

    def calculate_dt(
        self,
        event: NUMPY_TORCH,
        reference_time: FLOAT_TORCH,
        time_period: Optional[FLOAT_TORCH] = None,
    ) -> NUMPY_TORCH:
        """Calculate dt.
        First, it operates `t - reference_time`. And then it operates normalization if
        self.normalize_t is True. `time_period` is effective when normalization.

        Args:
            event (NUMPY_TORCH): [(b,) n, 4]
            reference_time (FLOAT_TORCH): The reference timestamp.
            time_period (Optional[FLOAT_TORCH], optional): If normalize is True, you can specify the
                period for the normalization. Defaults to None (normalize so that the max - min = 1).

        Returns:
            NUMPY_TORCH: dt array. [(b,) n]
        """
        dt = event[..., 2] - reference_time
        if self.normalize_t:  # to [0, 1]
            if time_period is None:
                time_period = nt_max(dt, -1) - nt_min(dt, -1)
            dt /= time_period[..., None]
        return dt

    # Functions for torch tensor
    # Functions for numpy array
    def warp_event_from_optical_flow(
        self, event: NUMPY_TORCH, flow: NUMPY_TORCH, reference_time: FLOAT_TORCH
    ) -> Tuple[NUMPY_TORCH, dict]:
        """Warp events from dense optical flow

        Args:
            event (np.ndarray) ... [(b,) n x 4]. Each event is (x, y, t, p)
            flow ... [(b,) 2, H, W]. Velocity (Optical flow) of the image plane at the position (x, y)
            reference_time (float) ... reference time

        Returns:
            warped_event (np.ndarray) ... [(b,) n, 4]. Warped event. (x', y', time, p). x' and y' could be float.
            feature (dict) ... Feature dict. if self.calculate_feature is True.
        """
        dt = self.calculate_dt(event, reference_time)

        if len(event.shape) == 2:
            event = event[None, ...]
            flow = flow[None, ...]
            dt = dt[None, ...]
        assert len(dt.shape) + 1 == len(flow.shape) - 1 == 3

        if is_numpy(event):
            assert is_numpy(flow) and is_numpy(dt)

            warped_numpy: np.ndarray = np.copy(event)
            nb = len(warped_numpy)
            _ix = event[..., 0].astype(np.int32)
            _iy = event[..., 1].astype(np.int32)
            for i in range(nb):
                warped_numpy[i, :, 0] = event[i, :, 0] - dt[i] * flow[i, 0, _ix[i], _iy[i]]
                warped_numpy[i, :, 1] = event[i, :, 1] - dt[i] * flow[i, 1, _ix[i], _iy[i]]

            warped_numpy[..., 2] = dt
            feat = self.feature_dense.calculate_feature(
                event.squeeze(), flow.squeeze(), skip=not self.calculate_feature
            )
            return warped_numpy.squeeze(), feat
        elif is_torch(event):
            assert is_torch(flow) and is_torch(dt)
            warped_torch = event.clone()
            flow_flat = flow.reshape((flow.shape[0], 2, -1))
            _ind = event[..., 0].long() * self.image_size[1] + event[..., 1].long()
            warped_torch[..., 0] = event[..., 0] - dt * torch.gather(flow_flat[:, 0], 1, _ind)
            warped_torch[..., 1] = event[..., 1] - dt * torch.gather(flow_flat[:, 1], 1, _ind)
            warped_torch[..., 2] = dt

            feat = self.feature_dense.calculate_feature(
                event, flow, skip=not self.calculate_feature
            )
            return warped_torch.squeeze(), feat

    def warp_event_2dof_xy(
        self,
        event: NUMPY_TORCH,
        translation: NUMPY_TORCH,
        reference_time: FLOAT_TORCH,
        time_period: Optional[FLOAT_TORCH] = None,
    ) -> Tuple[NUMPY_TORCH, dict]:
        """Warp events from simple 2DoF motion model, in the direction of x- and y- translation.
        During the warp, time period is normalized to [0, 1], if normalize_t is True.

        Args:
            event ... [4] or [n_event, 4]. each event has (x, y, t, p)
            translation (ndarray) ... [2, ] , [trans-x, trans-y] (in pixel)
            reference_time (float) ... reference time (timestamp of the reference frame)
            time_period (float) ... Time period of the event batch. Effective when normalize_t is True.

        Returns:
            warped_event (np.ndarray) ... [n, 4]. Warped event. (x', y', time, p). x' and y' could be float.
            feature (dict) ... Feature dict. if self.calculate_feature is True.
        """
        warped_event: NUMPY_TORCH
        if len(event.shape) == 1:
            event = event[None, :]  # now it's [1 x 4]
        dt = self.calculate_dt(event, reference_time, time_period)
        deltax = dt * translation[0]
        deltay = dt * translation[1]

        if is_numpy(event):
            assert is_numpy(translation) and is_numpy(dt)
            warped_event = np.vstack(
                [event[:, 0] + deltax, event[:, 1] + deltay, dt, event[:, 3]]
            ).T  # -1 (from translation pose into flow) * -1 (from warp, -dt) is plus.
            feat = self.feature_2dof.calculate_feature(dt, skip=not self.calculate_feature)
        elif is_torch(event):
            assert is_torch(translation) and is_torch(dt)
            warped_event = torch.vstack(
                [event[:, 0] + deltax, event[:, 1] + deltay, dt, event[:, 3]]
            ).T
            feat = self.feature_2dof.calculate_feature(dt, skip=not self.calculate_feature)
        return warped_event, feat
