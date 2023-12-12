import glob
import logging
import os
from typing import Tuple

import cv2
import numpy as np

from .base import DataLoaderBase

logger = logging.getLogger(__name__)


class HeliumDataLoader(DataLoaderBase):
    """Dataloader class for Helium dataset."""

    NAME = "HELIUM"

    def __init__(self, config: dict = {}):
        super().__init__(config)
        self.fps = 1000.0

    def __len__(self):
        # This is incompatible API, other loader returns number of events but
        # it returns nnumber of images.
        return self.num_images

    @property
    def num_images(self):
        return len(self.dataset_files["target_image"])

    # Main functions
    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `slider_depth`.

        Returns
           sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        data_path = os.path.join(self.dataset_dir, sequence_name)
        image_list = glob.glob(os.path.join(data_path, "*.tif"))
        image_list.sort()

        background_image = image_list[-1]
        target_image = image_list[:-1]

        sequence_file = {
            "background_image": background_image,
            "target_image": target_image,
        }
        return sequence_file

    def load_event(self, start_index: int, end_index: int, *args, **kwargs) -> np.ndarray:
        e = "This dataset has no event."
        logger.error(e)
        raise NotImplementedError(e)

    def load_image(self, index: int) -> Tuple[np.ndarray, float]:
        """Load image file and its timestamp

        Args:
            index (int): index of the image.

        Returns:
            Tuple[np.ndarray, float]: (image, timestamp)
        """
        if index == 0:  # background, base image
            image = cv2.imread(self.dataset_files["background_image"], cv2.IMREAD_GRAYSCALE)
            return image, 0.0
        image = cv2.imread(self.dataset_files["target_image"][index - 1], cv2.IMREAD_GRAYSCALE)
        ts = index / self.fps
        return image, ts

    def load_calib(self) -> dict:
        e = "Not supported!"
        logger.warning(e)
        return {"K": None, "D": None}

    def undistort(self, event_batch: np.ndarray) -> np.ndarray:
        e = "Not supported!"
        logger.error(e)
        raise NotImplementedError
