import logging
from typing import Optional, Tuple

import numpy as np

from ..types import FLOAT_TORCH, NUMPY_TORCH, is_numpy, is_torch

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    e = "Torch is disabled."
    logger.warning(e)


# Simulator module
def generate_events(
    n_events: int,
    height: int,
    width: int,
    tmin: float = 0.0,
    tmax: float = 0.5,
    dist: str = "uniform",
) -> np.ndarray:
    """Generate random events.

    Args:
        n_events (int) ... num of events
        height (int) ... height of the camera
        width (int) ... width of the camera
        tmin (float) ... timestamp min
        tmax (float) ... timestamp max
        dist (str) ... currently only "uniform" is supported.

    Returns:
        events (np.ndarray) ... [n_events x 4] numpy array. (x, y, t, p)
            x indicates height direction.
    """
    x = np.random.randint(0, height, n_events)
    y = np.random.randint(0, width, n_events)
    t = np.random.uniform(tmin, tmax, n_events)
    t = np.sort(t)
    p = np.random.randint(0, 2, n_events)

    events = np.concatenate([x[..., None], y[..., None], t[..., None], p[..., None]], axis=1)
    return events


# Batch processing
def reverse_event(event: NUMPY_TORCH) -> NUMPY_TORCH:
    """Reverse event data.
    It reverses the timestamp and the polarity.

    Args:
        event (NUMPY_TORCH) ... [n x 4]
    Returns:
        rev_event (NUMPY_TORCH) ... [n x 4]
    """
    ts = event[:, 2]
    if is_numpy(event):
        rev_event = np.copy(event)
        rev_event[:, 2] = np.abs(np.max(ts) - ts) + np.min(ts)
    elif is_torch(event):
        rev_event = event.clone()
        rev_event[:, 2] = torch.abs(ts.max() - ts) + ts.min()
    rev_event[:, 3] *= -1
    return sort_event_by_timestamp(rev_event)


def sort_event_by_timestamp(events: NUMPY_TORCH) -> NUMPY_TORCH:
    """Sort events by timestamp

    Args:
        events (NUMPY_TORCH): [n, 4]

    Returns:
        NUMPY_TORCH: [n, 4]
    """
    return events[events[:, 2].argsort()]


def filter_event(
    events: NUMPY_TORCH, start_time: Optional[float] = None, end_time: Optional[float] = None
) -> NUMPY_TORCH:
    """Filter event based on timestamps.
    event should be sorted based on time.

    Args:
        events (NUMPY_TORCH): [n, 4]
        start_time (Optional[float], optional): _description_. Defaults to None.
        end_time (Optional[float], optional): _description_. Defaults to None.

    Returns:
        NUMPY_TORCH: Filered events
    """
    if start_time is None and end_time is None:
        raise ValueError("Either start_time or end_time should be non-None")

    i1 = np.searchsorted(events[:, 2], start_time) if start_time is not None else 0
    i2 = np.searchsorted(events[:, 2], end_time) if end_time is not None else len(events)

    if i1 >= i2 or i1 >= len(events):
        # logger.warning('No events filtered')
        return np.array([])
    return events[i1:i2]


def crop_event(events: NUMPY_TORCH, x0: int, x1: int, y0: int, y1: int) -> NUMPY_TORCH:
    """Crop events.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the crop, at row[0]
        x1 (int): End of the crop, at row[0]
        y0 (int): Start of the crop, at row[1]
        y1 (int): End of the crop, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    mask = (
        (x0 <= events[..., 0])
        * (events[..., 0] < x1)
        * (y0 <= events[..., 1])
        * (events[..., 1] < y1)
    )
    cropped = events[mask]
    return cropped



def remove_event(events: NUMPY_TORCH, x0: int, x1: int, y0: int, y1: int) -> NUMPY_TORCH:
    """Remove events in specific area.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the window, at row[0]
        x1 (int): End of the window, at row[0]
        y0 (int): Start of the window, at row[1]
        y1 (int): End of the window, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    mask = (
        (x0 <= events[..., 0])
        * (events[..., 0] < x1)
        * (y0 <= events[..., 1])
        * (events[..., 1] < y1)
    )
    cropped = events[~mask]
    return cropped


def search_exact_event(events: NUMPY_TORCH, x: int, y: int) -> NUMPY_TORCH:
    """Get events at the same position.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x (int): Position at row[0]
        y (int): Position at row[1]

    Returns:
        NUMPY_TORCH: List of events at [x, y].
    """
    mask = (x == events[..., 0]) * (y == events[..., 1])
    cropped = events[mask]
    return cropped


def shift_event(events: NUMPY_TORCH, x0: int, y0: int) -> NUMPY_TORCH:
    """Shift events.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the shift, at row[0]
        y0 (int): Start of the shift, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    return events + np.array([x0, y0, 0, 0])


def random_sample_event(events: NUMPY_TORCH, percentage: float) -> NUMPY_TORCH:
    """Randomly sample n percent of event data.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        percentage (float)

    Returns:
        NUMPY_TORCH: Sampled events. [(n x percentage / 100), 4].
    """
    assert percentage <= 100
    sample_indice = np.random.permutation(np.arange(len(events)))[
        : int(len(events) * percentage / 100)
    ]
    sample_indice = np.sort(sample_indice)
    return sort_event_by_timestamp(events[sample_indice])


def set_event_origin_to_zero(events: np.ndarray, x0: int, y0: int, t0: float=0.0) -> np.ndarray:
    """Set each origin of each row to 0.

    Args:
        events (np.ndarray): [n x 4]. [x, y, t, p].
        x0 (int): x origin
        y0 (int): y origin
        t0 (float): t origin

    Returns:
        np.ndarray: [n x 4]. x is in [0, xmax - x0], and so on.
    """
    basis = np.array([x0, y0, t0, 0.])
    if is_torch(events):
        basis = torch.from_numpy(basis)
    return events - basis


def normalize_time(events: NUMPY_TORCH) -> Tuple[NUMPY_TORCH, float]:
    """Normalize time to [0, 1].

    Args:
        events (np.ndarray): [n x 4]. [x, y, t, p].

    Returns:
        np.ndarray: [n x 4]. x is in [0, xmax - x0], and so on.
        float: Absolute time scale.
    """
    if is_numpy(events):
        time_scale = np.max(events[:, 2]) - np.min(events[:, 2])
        events[:, 2] = (events[:, 2] - np.min(events[:, 2])) / time_scale
    elif is_torch(events):
        time_scale = torch.max(events[:, 2]) - torch.min(events[:, 2])
        events[:, 2] = (events[:, 2] - torch.min(events[:, 2])) / time_scale

    return events, time_scale


def undistort_events(events, map_x, map_y, h, w):
    """Undistort (rectify) events.
    Args:
        events ... [x, y, t, p]. X is height direction.
        map_x, map_y... meshgrid

    Returns:
        events... events that is in the camera plane after undistortion.
    TODO check overflow
    """
    # k = np.int32(map_y[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # l = np.int32(map_x[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # k = np.int32(map_y[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # l = np.int32(map_x[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # undistort_events = np.copy(events)
    # undistort_events[:, 0] = l
    # undistort_events[:, 1] = k
    # return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]

    k = np.int32(map_y[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    l = np.int32(map_x[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    undistort_events = np.copy(events)
    undistort_events[:, 0] = k
    undistort_events[:, 1] = l
    return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]


def split_polarity(events: NUMPY_TORCH, keep_size: bool = False) -> Tuple[NUMPY_TORCH, NUMPY_TORCH]:
    """Crop events.

    Args:
        events (NUMPY_TORCH): (b,) n, 4. The last col is pol.
        keep_size (bool, optional): If true, returns the multiplied result. If False, masked resuilt.
        Defaults to False.

    Returns:
        (NUMPY_TORCH, NUMPY_TORCH): positive and negative events.
    """
    mask = 0 < events[..., 3]
    if keep_size:
        positive = events * mask
        negative = events * (not mask)
    else:
        positive = events[mask]
        negative = events[~mask]
    return positive, negative


# Voxel conversion
def create_event_voxel(
    x: torch.Tensor,
    y: torch.Tensor,
    pol: torch.Tensor,
    time: torch.Tensor,
    voxel_shape: tuple,
    normalize: bool = False,
):
    """Create voxel grid with weights.
    Original code is https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/representations.py
    This encode positive and negative events together.
    The polarity information is used for the weight of the voxel.

    Args:
        x (torch.Tensor) ... (n_events, ). x is width direction.
        y (torch.Tensor) ... (n_events, ).
        pol (torch.Tensor) ... (n_events, ). The polarity is [-1, +1].
        time (torch.Tensor) ... (n_events, ).
        voxel_shape (tuple) ... [C, H, W].
        normalize (bool) ... True to normalie the output voxel.

    Returns:
        voxel_grid (torch.Tensor) ... (voxel_shape).

    """
    assert x.shape == y.shape == pol.shape == time.shape
    assert x.ndim == 1

    C, H, W = voxel_shape
    with torch.no_grad():
        voxel_grid = x.new_zeros(voxel_shape, dtype=torch.double)

        t_norm = time
        t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

        x0 = x.int()  # int() gives floor
        y0 = y.int()
        t0 = t_norm.int()

        # value = 2 * pol - 1   # for pol in [0, 1]
        value = pol  # for pol already [-1, 1]

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                for tlim in [t0, t0 + 1]:

                    mask = (
                        (xlim < W)
                        & (xlim >= 0)
                        & (ylim < H)
                        & (ylim >= 0)
                        & (tlim >= 0)
                        & (tlim < C)
                    )
                    interp_weights = (
                        value
                        * (1 - (xlim - x).abs())
                        * (1 - (ylim - y).abs())
                        * (1 - (tlim - t_norm).abs())
                    )

                    index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        if normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

    return voxel_grid


# EVFlownet utils
def calc_floor_ceil_delta(x):
    """
    Args:
        x (torch.Tensor)

    Returns
        [floor(x), (floor(x) + 1) - x], [ceil(x), x - floor(x)]
    """
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]


def create_update(x, y, t, dt, p, vol_size: tuple):
    """Helper function to create discretized event volume.

    Args:
        x, y, t (torch.Tensor)
        vol_size (tuple) ... (b, x, y). x is height.
    """
    # This is old, when x-width and y-height.
    # assert (x >= 0).byte().all() and (x < vol_size[2]).byte().all()
    # assert (y >= 0).byte().all() and (y < vol_size[1]).byte().all()
    assert (x >= 0).byte().all() and (x < vol_size[1]).byte().all()
    assert (y >= 0).byte().all() and (y < vol_size[2]).byte().all()
    assert (t >= 0).byte().all() and (t < vol_size[0] // 2).byte().all()

    vol_mul = torch.where(
        p < 0,
        torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
        torch.zeros(p.shape, dtype=torch.long),
    )
    # This is old, when x-width and y-height.
    # inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) + (vol_size[2]) * y + x
    inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) + (vol_size[2]) * x + y
    vals = dt
    return inds, vals


def generate_discretized_event_volume(events: torch.Tensor, vol_size: tuple):
    """Create discretized event volume for a given patch.
    Original code is https://github.com/alexzzhu/EventGAN/blob/master/EventGAN/utils/event_utils.py
    This volume encode positive and negative polarity separately.
    This means, [:n_bin // 2] ... Positive, [n_bin // 2:] negative events aggregation.

    Args:
        events (torch.Tensor) ... [n_events, 4]. 4 is [x, y, t, p]
        vol_size (tuple) ... tuple specifing the return volume size.

    Returns:
        volume (torch.tensor) ... [t, x, y]. t is discretized.
    """
    volume = events.new_zeros(vol_size)
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] // 2 - 1) / (t_max - t_min))
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    inds_fl, vals_fl = create_update(x, y, ts_fl[0], ts_fl[1], events[:, 3], vol_size)
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)
    inds_ce, vals_ce = create_update(x, y, ts_ce[0], ts_ce[1], events[:, 3], vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume
