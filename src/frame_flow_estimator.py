import json
import logging
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")

import logging
import os

# from pivpy import io

from . import data_loader, solver, utils, visualizer

logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


class FrameFlowEstimator(object):
    def __init__(self, visualizer_module: visualizer.Visualizer) -> None:
        self.visualizer = visualizer_module

    def estimate(self, method: str, frame0: np.ndarray, frame1: np.ndarray, frame2: np.ndarray, config: dict):
        if method == 'opencv_flow':
            params = config["params_opencv_flow"]
            return self.opencv_farneback(frame1, frame2, params, visualize_frame=False)
        elif method == 'opencv_flow_two_steps':
            params = config["params_opencv_flow"]
            return self.opencv_farneback_two_step(frame0, frame1, frame2, params)
        elif method == 'openpiv':
            params = config["params_openpiv"]
            return self.consecutive_openpiv(frame1, frame2, params, visualize_frame=False)
        e = f"{method} is not supported"
        logger.error(e)
        raise NotImplementedError(e)

    # Two-steps: requires background frame and two consective frames
    def opencv_farneback_two_step(
            self, frame0, frame1, frame2, params_opencv_flow,
    ) -> np.ndarray:
        """Estimate optical flow using OpenCV function.

        Args:
            frame1 (np.ndarray): _description_
            frame2 (np.ndarray): _description_
            params_opencv_flow (_type_): _description_
            viz (_type_): _description_
            visualize_frame (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        f01 = self.opencv_farneback(frame0, frame1, params_opencv_flow, visualize_frame=False)
        f02 = self.opencv_farneback(frame0, frame2, params_opencv_flow, visualize_frame=False)

        p01 = utils.standardize_image_center(utils.poisson_reconstruct(f01[1], f01[0], np.zeros_like(f01[0]))).astype(np.uint8)
        p02 = utils.standardize_image_center(utils.poisson_reconstruct(f02[1], f02[0], np.zeros_like(f02[0]))).astype(np.uint8)

        f12 = utils.bos_optical_flow(p01, p02, params_opencv_flow).transpose(2, 0, 1)  # 2 x H x W
        return f12

    # One-step: requires only consective frames
    def opencv_farneback(
            self, frame1, frame2, params_opencv_flow, visualize_frame=False
    ) -> np.ndarray:
        """Estimate optical flow using OpenCV function.
        This gives forward optical flow.  TODO confirm it!

        Args:
            frame1 (np.ndarray): _description_
            frame2 (np.ndarray): _description_
            params_opencv_flow (_type_): _description_
            viz (_type_): _description_
            visualize_frame (bool, optional): _description_. Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        f_next = utils.bos_optical_flow(frame1, frame2, params_opencv_flow)  # concurrent frames
        if visualize_frame:
            self.visualizer.visualize_optical_flow(
                f_next[..., 0], f_next[..., 1], file_prefix="frame_flow_concurrent"
            )
            self.visualizer.visualize_image(frame1, file_prefix="frame_current")
            self.visualizer.visualize_image(frame2, file_prefix="frame_next")
        crop_flow = f_next.transpose(2, 0, 1)  # 2 x H x W
        pad_flow = utils.pad_to_same_resolution(crop_flow, params_opencv_flow, 0)
        return pad_flow
