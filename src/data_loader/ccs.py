import csv
import glob
import logging
import os
import pathlib
from typing import Tuple

import cv2
import h5py
import numpy as np

from src.utils import check_key_and_bool, extract_mp4

from .base import DataLoaderBase

logger = logging.getLogger(__name__)


logger.warning("Due to its unstability, we don't use Metavision directly from the script. Please convert .raw to hdf5.")  # type: ignore
OPENEB_ENABLED = False


IMG_FORMATS = [
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
]  # include image suffixes


def load_frame_timestamps(path):
    try:
        frame_timestamps = np.loadtxt(path, dtype=int)  # delimiter is space
        frame_timestamps = frame_timestamps[frame_timestamps[:, 2] == 1]  # only positive edges
        return frame_timestamps[:, 0]
    except ValueError:
        e = "Trying another format reading.."
        logger.warning(e)
        # Newer version version of Metavision
        frame_timestamps = np.loadtxt(path, dtype=int, delimiter=",")
        frame_timestamps = frame_timestamps[frame_timestamps[:, 0] == 1]  # only positive edges
        return frame_timestamps[:, 2]


def h5py_loader(path: str) -> dict:
    """Basic loader for .hdf5 files.
    Args:
        path (str) ... Path to the .hdf5 file.

    Returns:
    """
    f = h5py.File(path, "r")
    if len(f["raw_events"]["t"]) > 2147483647:  # larger than int32
        w = "Please check the size of the data and data type. int32 may not be enough."
        logger.warning(w)
    data = {
        "x": np.array(f["raw_events"]["x"], dtype=np.int16),
        "y": np.array(f["raw_events"]["y"], dtype=np.int16),
        "t": np.array(f["raw_events"]["t"], dtype=np.int32),
        "p": np.array(f["raw_events"]["p"], dtype=bool),
    }
    # 'gray_ts': np.array(data['davis']['right']['image_raw_ts'], dtype=np.float64)
    f.close()
    return data


class CcsDataLoader(DataLoaderBase):
    """Dataloader class for the co-capture system."""

    NAME = "CCS"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        # Cache because the dataset is only serial loading text file.
        self._time_cache = None
        self._len_cache = None
        self._image_cache = None
        self._len_image = None
        self.raw_reader = None
        self.warp_frame = check_key_and_bool(config, "warp")
        logger.info(f"Warp frames according to event camera: {self.warp_frame}")

    def __len__(self):
        if self._len_cache is None:
            self.set_len_cache()
        return self._len_cache

    @property
    def num_images(self):
        if self._len_image is None:
            self.set_image_cache()
        return self._len_image

    @property
    def num_thermals(self):
        return len(self.dataset_files["thermal"])

    def set_raw_reader(self):
        assert OPENEB_ENABLED
        from metavision_core.event_io import RawReader

        self._max_events = 100000000
        self.raw_reader = RawReader(self.dataset_files["event_raw"], max_events=self._max_events)

    # Helper - cache functions
    def clear_time_cache(self):
        self._time_cache = None

    def clear_len_cache(self):
        self._len_cache = None

    def set_len_cache(self):
        if OPENEB_ENABLED:
            self.set_raw_reader()
            if self._time_cache is None:
                time_list = np.zeros((200000000,), dtype=np.float64)
            cnt = 0
            while not self.raw_reader.is_done():
                events = self.raw_reader.load_n_events(100000)
                time_list[cnt : cnt + len(events)] = events["t"] / 1e6
                cnt += len(events)
            self._len_cache = cnt - 1
            if self._time_cache is None:
                self._time_cache = time_list[:cnt]
            self.set_raw_reader()
        else:
            self._len_cache = len(self.event_data["x"])
            if self._time_cache is None:
                self._time_cache = self.event_data["t"] / 1e6

    def set_image_cache(self):
        data_path = self.dataset_files["frame"]
        suffix = pathlib.Path(data_path).suffix
        frame_dir = f"{pathlib.Path(data_path).parents[0]}/frames"

        if suffix == ".mp4" and not os.path.isdir(frame_dir):
            pathlib.Path(frame_dir).mkdir()
            extract_mp4(data_path, frame_dir)

        files = sorted(glob.glob(os.path.join(frame_dir, "*.*")))
        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]

        timestamp_path = self.dataset_files["event_trigger"]
        timestamps = load_frame_timestamps(timestamp_path) / 1e6

        self._image_cache = {"image": images, "timestamp": timestamps}
        if self.warp_frame:
            self._image_cache["homography"] = np.loadtxt(self.dataset_files["homography"])

        self._len_image = len(images)
        logger.info(f"Num images {self._len_image}")

    # Main functions
    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `slider_depth`.

        Returns
           sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        data_path = os.path.join(self.dataset_dir, sequence_name)
        # events
        event_path = os.path.join(data_path, "prophesee_0")
        event_raw_file = os.path.join(event_path, "cd_events.raw")
        event_hdf5_file = os.path.join(event_path, "events.hdf5")
        event_file = os.path.join(event_path, "cd.csv")
        event_trigger_file = os.path.join(event_path, "trigger_events.txt")
        event_roi_csv = os.path.join(event_path, "roi.csv")
        # frames
        frame_path = os.path.join(data_path, "basler_0")
        frame_file = os.path.join(frame_path, "frames.mp4")
        frame_2x_file = os.path.join(frame_path, "frames_2X_240fps.mp4")  # upsampled video if any
        homography_file = os.path.join(data_path, "homography.txt")
        # frame_config_file = os.path.join(frame_path, "config.yaml")

        # thermal
        thermal_path = os.path.join(data_path, "thermal")
        thermal_files = sorted(glob.glob(os.path.join(thermal_path, "*.csv")))
        sequence_file = {
            "event_raw": event_raw_file,
            "event_hdf": event_hdf5_file,
            "event_csv": event_file,
            "event_trigger": event_trigger_file,
            "event_roi": event_roi_csv,
            "frame": frame_file,
            "frame_2x": frame_2x_file,
            "homography": homography_file,
            "thermal": thermal_files,
        }
        return sequence_file

    def set_sequence(self, sequence_name: str, undistort: bool = False) -> None:
        super().set_sequence(sequence_name)

        if not OPENEB_ENABLED:  # Assume hdf5-based loading
            logger.info("Set up hdf5 loader instead of Metavision SDK.")
            self.event_data = h5py_loader(self.dataset_files["event_hdf"])
            # Setting up time suration statistics
            self.min_ts = self.event_data["t"].min() / 1e6
            self.max_ts = self.event_data["t"].max() / 1e6
            self.data_duration = self.max_ts - self.min_ts

        if os.path.exists(self.dataset_files["event_roi"]):
            logger.info("Theere is a crop file from the recording.")
            try:
                self.crop_info = self.load_recording_cropinfo(self.dataset_files["event_roi"])
            except:
                logger.warning("Attention! Failed to load the ROI info.")

    def load_recording_cropinfo(self, csv_file: str):
        """Load csv texrt about the ROI information.
        Each row of the ROI file is [y0 (in OUR coordinate), x0 (in OUR coordinate), width, height]
        TODO not tested it!

        Args:
            csv_file (str): _description_
        """
        rois = np.loadtxt(csv_file, delimiter=",")
        if len(rois.shape) == 1:
            rois = rois[None]
        rois_our_coord = np.zeros_like(rois)
        rois_our_coord[:, 0] = rois[:, 1]  # x0
        rois_our_coord[:, 1] = rois[:, 1] + rois[:, 3]  # x1 = x0 + height
        rois_our_coord[:, 2] = rois[:, 0]  # y0
        rois_our_coord[:, 3] = rois[:, 0] + rois[:, 2]  # y1 = y0 + width
        logger.warning("The ROI coordinate is not tested enough! Please be careful.")
        return rois_our_coord

    def load_event(self, start_index: int, end_index: int, *args, **kwargs) -> np.ndarray:
        """Load events.
        The data format is comma-separated .csv file.
        This is the outout of the standalone converter of OpenEB .raw data format.
        For more details, please see https://github.com/prophesee-ai/openeb/blob/8e57704b195d653bdd2075a0d48b6539e159d976/standalone_samples/metavision_evt3_raw_file_decoder

        The data format is `x,y,polarity,t`.
            x means in width direction, and y means in height direction.
            polarity is 0 or 1. t is longint [unit?].

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height (OpenCV).
            t is absolute value in second. p is [-1, +1].
        """
        if end_index > len(self):
            e = f"Specified {start_index} to {end_index} index, but there are only {len(self)} events."
            logger.error(e)
            raise IndexError(e)
        if OPENEB_ENABLED:
            if self.raw_reader is None:
                self.set_raw_reader()
            events = self.load_event_from_raw(start_index, end_index)
        else:
            events = self.load_event_from_hdf(start_index, end_index)

        n_events = end_index - start_index
        if len(events) == 0:
            e = f"Specified {start_index} to {end_index} index, but no events."
            logger.error(e)
            raise IndexError(e)
        elif len(events) < n_events:
            e = f"Specified {start_index} to {end_index} index, but less events."
            logger.warning(e)

        # if self.auto_undistort:
        #     events = self.undistort(events)
        return events

    def load_event_from_hdf(self, start_index: int, end_index: int) -> np.ndarray:
        """Load events from HDF5 file.
        Please convert .raw file into .hdf5 file using [scripts/convert_raw_to_hdf5.py].

        Args:
            start_index (int): _description_
            end_index (int): _description_

        Raises:
            IndexError: _description_

        Returns:
            np.ndarray: _description_
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)
        if len(self) <= start_index:
            logger.error(f"Specified {start_index} to {end_index} index for {len(self)}.")
            raise IndexError
        events[:, 0] = self.event_data["y"][start_index:end_index]
        events[:, 1] = self.event_data["x"][start_index:end_index]
        events[:, 2] = self.event_data["t"][start_index:end_index] / 1e6  # from micro sec to sec
        events[:, 3] = self.event_data["p"][start_index:end_index]
        return events

    def load_event_from_raw(self, start_index: int, end_index: int) -> np.ndarray:
        """Load events from .raw file.

        Args:
            start_index (int): _description_
            end_index (int): _description_

        Raises:
            IndexError: _description_

        Returns:
            np.ndarray: _description_
        """
        self.raw_reader.reset()
        self.raw_reader.seek_event(start_index)
        n_events = end_index - start_index
        events = self.raw_reader.load_n_events(n_events)
        events = np.stack([events["y"], events["x"], events["t"] / 1e6, events["p"]]).T
        return events

    def index_to_time(self, index: int) -> float:
        """Event index to time.

        Args:
            index (int): index of event

        Returns:
            float: time in sec
        """
        if self._time_cache is None:
            self.set_len_cache()
        return self._time_cache[index]

    def image_index_to_time(self, index: int) -> float:
        """Image index to time.

        Args:
            index (int): index of image

        Returns:
            float: time in sec
        """
        if self._image_cache is None:
            self.set_image_cache()
        return self._image_cache["timestamp"][index]

    def time_to_index(self, time: float) -> int:
        """Time to event index.

        Args:
            time (float): time in sec

        Returns:
            int: index
        """
        if self._time_cache is None:
            self.set_len_cache()
        ind = np.searchsorted(self._time_cache, time)
        return ind - 1

    def time_to_image_index(self, time: float) -> int:
        """Time to image index

        Args:
            time (float): time in sec

        Returns:
            int: index
        """
        if self._image_cache is None:
            self.set_image_cache()
        ind = np.searchsorted(self._image_cache["timestamp"], time)
        return ind - 1

    def load_image(self, index: int) -> Tuple[np.ndarray, float]:
        """Load image file and it's timestamp

        Args:
            index: index of the image
            warp: warp the image with given homography

        Returns:
            Tuple[np.ndarray, float]: (image, timestamp)
        """
        if self._image_cache is None:
            self.set_image_cache()
        assert index < self._len_image

        image_path = self._image_cache["image"][index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        timestamp = self._image_cache["timestamp"][index]

        if self.warp_frame:
            image = cv2.warpPerspective(
                image, self._image_cache["homography"], (self._WIDTH, self._HEIGHT)
            )
        return image, timestamp

    def load_thermal(self, index: int) -> Tuple[np.ndarray, float]:
        """Load thermal recording data.

        Args:
            index: index of the frame.
            NOTE: this is not sycned.

        Returns:
            Tuple[np.ndarray, float]: (image, timestamp)
        """
        assert index < self.num_thermals
        thermal_file = self.dataset_files["thermal"][index]
        file = open(thermal_file, "r")

        thermal_array = []
        cnt = 0
        while 1:
            lines = file.readlines()
            if not lines:
                break
            for line in lines:
                row = [float(i) for i in line.split(",") if i != "\n"]
                cnt += 1
                thermal_array.append(row)

        thermal_array = np.array(thermal_array)
        assert len(thermal_array.shape) == 2
        return thermal_array

    def load_calib(self) -> dict:
        """Load calibration file.

        Outputs:
            (dict) ... {"K": camera_matrix, "D": distortion_coeff}
                camera_matrix (np.ndarray) ... [3 x 3] matrix.
                distortion_coeff (np.array) ... [5] array.
        """
        e = "Not supported!"
        logger.warning(e)
        # raise NotImplementedError
        return {"K": None, "D": None}

    def undistort(self, event_batch: np.ndarray) -> np.ndarray:
        """Undistort events.

        Args:
            event_batch (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        e = "Not supported!"
        logger.error(e)
        raise NotImplementedError
