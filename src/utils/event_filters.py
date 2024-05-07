import logging

import numpy as np

from ..event_image_converter import EventImageConverter
from ..types import NUMPY_TORCH
from .event_utils import crop_event, set_event_origin_to_zero, search_exact_event
from .misc import check_key_and_bool

DEFAULT_INDEX_CONVENTION = {"x": 0, "y": 1, "t": 2, "p": 3}

logger = logging.getLogger(__name__)


def background_activity_filter(
    events: NUMPY_TORCH,
    image_shape: tuple,
    dt: float,
    ksize: int = 1,
    num_support_event: int = 1,
    index_convention: dict = None,
) -> NUMPY_TORCH:
    """Background Activity Filter in Tobi 2008 "Frame-Free Dynamic Digital Vision."
    https://github.com/SensorsINI/jaer/blob/master/src/net/sf/jaer/eventprocessing/filter/BackgroundActivityFilter.java
    Legacy wrapper of continuous_activity_filter. This function will instantiate new time map for
    every event batch. For using an already existing time map from previous event batches, use
    continuous_background_activity_filter

    Args:
        events:
        image_shape:
        dt:
        ksize:
        num_support_event:
        index_convention:

    Returns:

    """
    filtered_event, _ = continuous_background_activity_filter(
        events, image_shape, dt, ksize, num_support_event, index_convention
    )
    return filtered_event


def continuous_background_activity_filter(
    events: NUMPY_TORCH,
    image_shape: tuple,
    dt: float,
    ksize: int = 1,
    num_support_event: int = 1,
    index_convention: dict = None,
    time_map: NUMPY_TORCH = None,
) -> NUMPY_TORCH:
    """Background Activity Filter in Tobi 2008 "Frame-Free Dynamic Digital Vision."
    https://github.com/SensorsINI/jaer/blob/master/src/net/sf/jaer/eventprocessing/filter/BackgroundActivityFilter.java

    Args:
        events: n x 4 array of events
        image_shape: (height, width)
        dt:
        ksize:
        num_support_event:
        index_convention:

    Returns:
        NUMPY_TORCH: Filtered events
    """
    if index_convention is None:
        index_convention = DEFAULT_INDEX_CONVENTION
    ix = index_convention["x"]
    iy = index_convention["y"]
    it = index_convention["t"]
    num_events = len(events)
    # Create time map
    if time_map is None:
        time_map = np.zeros(image_shape, dtype=np.float64)
    filtered_event = []
    for e in events:
        x, y = int(e[ix]), int(e[iy])
        ts = e[it]
        time_map[x, y] = max(time_map[x, y], ts)
        xmin, ymin = max(0, x - ksize), max(0, y - ksize)
        xmax, ymax = min(image_shape[0], x + ksize + 1), min(image_shape[1], y + ksize + 1)
        # if (filterHotPixels && xx == x && yy == y) {
        #     continue; // like BAF, don't correlate with ourself
        # }
        time_array = np.sort(time_map[xmin:xmax, ymin:ymax].reshape(-1))
        # print(time_array)
        last_timestamp = time_array[-1 - num_support_event]
        delta_t = ts - last_timestamp
        if delta_t < dt:
            filtered_event.append(e)
    # logger.debug(f'BAF removed {100 * (1 - len(filtered_event) / len(events)):5.2f} % of the events')
    if len(filtered_event) == 0:
        return np.array([]), time_map
    return np.vstack(filtered_event), time_map


def hot_pixel_filter(
    events: NUMPY_TORCH, image_shape: tuple, hot_pixel: int = 10, index_convention: dict = None
) -> NUMPY_TORCH:
    """

    Args:
        events (NUMPY_TORCH):
        image_shape:
        hot_pixel:
        index_convention:

    Returns:
        NUMPY_TORCH: Filtered events
    """
    if index_convention is None:
        index_convention = DEFAULT_INDEX_CONVENTION
    ix = index_convention["x"]
    iy = index_convention["y"]
    # Create time map
    imager = EventImageConverter(image_shape)
    iwe = imager.create_iwe(events, sigma=0)
    indices = np.where(iwe > hot_pixel)
    indices = [[x, y] for x, y in zip(*indices)]
    filtered_event = []
    for e in events:
        x, y = int(e[ix]), int(e[iy])
        if [x, y] not in indices:
            filtered_event.append(e)
    return np.vstack(filtered_event)


def flicker_filter(
    events: NUMPY_TORCH,  dt: float = 0.01
) -> NUMPY_TORCH:
    """
    Args:
        events (NUMPY_TORCH):
        image_shape:
        hot_pixel:
        index_convention:

    Returns:
        NUMPY_TORCH: Filtered events
    """
    is_linked = [False for _ in range(len(events))]
    for i, e in enumerate(events):
        searched_epp, mask = search_exact_event(events, e[0], e[1])
        for j in range(0, len(searched_epp) - 1):
            if searched_epp[j, 3] !=  searched_epp[j + 1, 3] and searched_epp[j, 2] > searched_epp[j + 1, 2] - dt:
                is_linked[np.where(mask)[0][j]] = True
                is_linked[np.where(mask)[0][j + 1]] = True
    return events[is_linked], events[~np.array(is_linked)]


class EventFilter:
    def __init__(self, image_shape, filter_config):
        self.image_shape = image_shape
        self.filter_params = filter_config["parameters"]

        if filter_config["filters"] is None:
            self.filters = []
        else:
            self.filters = filter_config["filters"]            
        if "xmin" in self.filter_params.keys():
            self.filters = ["CROP"] + self.filters
        if "index_convention" in filter_config.keys():
            self.index_convention = filter_config["index_convention"]
        else:
            self.index_convention = DEFAULT_INDEX_CONVENTION
        self.continuous_update = check_key_and_bool(self.filter_params, "BAF_continuous_update")
        self.time_map = None
        self.setup()

    def setup(self):
        FILTER_SET = {
            "BAF": self.background_activity_filter,
            "HOT": self.hot_pixel_filter,
            "CROP": self.crop,
        }
        self.filter_func = [FILTER_SET[f] for f in self.filters]
        logger.info(f"Setup filters: {self.filter_func} \n with parameters: {self.filter_params}")

    def process(self, events: NUMPY_TORCH):
        for i, f in enumerate(self.filter_func):
            num_events = len(events)
            if num_events < 10:
                logger.warning("Too small events after filering.")
                return events
            events = f(events)
            logger.debug(
                f"{self.filters[i]} removed {100 * (1 - len(events) / num_events):5.2f} % of the events (Originally {num_events})"
            )
        return events

    def crop(self, events: NUMPY_TORCH):
        cropped_events = crop_event(
            events,
            self.filter_params["xmin"],
            self.filter_params["xmax"],
            self.filter_params["ymin"],
            self.filter_params["ymax"],
        )
        return cropped_events

    def background_activity_filter(self, events: NUMPY_TORCH):
        filtered_events, self.time_map = continuous_background_activity_filter(
            events,
            self.image_shape,
            self.filter_params["BAF_dt"],
            self.filter_params["BAF_ksize"],
            self.filter_params["BAF_num_support_event"],
            index_convention=self.index_convention,
            time_map=self.time_map,
        )
        if not self.continuous_update:
            self.time_map = None
        return filtered_events

    def hot_pixel_filter(self, events: NUMPY_TORCH):
        return hot_pixel_filter(
            events,
            self.image_shape,
            self.filter_params["HOT_thresh"],
            index_convention=self.index_convention,
        )
