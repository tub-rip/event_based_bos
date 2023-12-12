import csv
import glob
import logging
import os
import pathlib
from typing import Tuple

import cv2
import numpy as np

from src.utils import extract_mp4

from .base import DataLoaderBase

logger = logging.getLogger(__name__)

IMG_FORMATS = ["png"]  # include image suffixes


class E2vidDataLoader(DataLoaderBase):
    """Dataloader class for frames reconstructed from e2vid"""

    NAME = "E2VID"

    def __init__(self, config: dict = {}, overwrite_dataset: bool = False):
        super().__init__(config)
        # Cache because the dataset is only serial loading text file.
        self._time_cache = None
        self._len_cache = None
        self._image_cache = None
        self._len_image = None
        self._do_overwrite_dataset = overwrite_dataset

    def __len__(self):
        if self._len_cache is None:
            self.set_len_cache()
        return self._len_cache

    @property
    def num_images(self):
        if self._len_image is None:
            self.set_image_cache()
        return self._len_image

    # Helper - cache functions
    def clear_time_cache(self):
        self._time_cache = None

    def clear_len_cache(self):
        self._len_cache = None

    def set_len_cache(self):
        if self._time_cache is None:
            time_list = np.zeros((200000000,), dtype=np.float64)
        cnt = 0
        file = open(self.dataset_files["event"], "r")
        while 1:
            lines = file.readlines()
            if not lines:
                break
            for line in lines:
                if self._time_cache is None:
                    val = line.split(",")
                    time_list[cnt] = np.float64(val[3])  # / 1e6
                cnt += 1
        self._len_cache = cnt - 1
        if self._time_cache is None:
            self._time_cache = time_list[:cnt]

    def set_image_cache(self):

        data_path = self.dataset_files["frame"]

        files = sorted(glob.glob(os.path.join(data_path, "*.*")))
        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]

        timestamp_path = self.dataset_files["timestamp"]
        timestamps = np.loadtxt(timestamp_path, dtype=float).astype(float)

        self._image_cache = {"image": images, "timestamp": timestamps}
        self._len_image = len(images)

    # Main functions
    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `slider_depth`.

        Returns
           sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        if self._do_overwrite_dataset:
            head_tail = os.path.split(self.dataset_dir)
            self.dataset_dir = os.path.join(head_tail[0], "E2VID")

        data_path = os.path.join(self.dataset_dir, sequence_name)
        frame_path = data_path
        timestamp_file = os.path.join(data_path, "timestamps.txt")

        sequence_file = {"frame": frame_path, "timestamp": timestamp_file}
        return sequence_file

    def index_to_time(self, index: int) -> float:
        raise NotImplementedError

    def time_to_index(self, time: float) -> int:
        if self._image_cache is None:
            self.set_image_cache()
        ind = np.searchsorted(self._image_cache["timestamp"], time)
        return ind - 1

    def load_image(self, index: int) -> Tuple[np.ndarray, int]:
        """Load image file and it's timestamp

        Args:
            index: of the image

        Returns:
            (image, timestamp)
        """
        if self._image_cache is None:
            self.set_image_cache()
        assert index < self._len_image

        image_path = self._image_cache["image"][index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        timestamp = self._image_cache["timestamp"][index]
        return image, timestamp

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
