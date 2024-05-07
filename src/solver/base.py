import logging
import math
import os
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np
import optuna
import scipy
import torch

from .. import costs, event_image_converter, utils, visualizer, warp
from ..types import NUMPY_TORCH

logger = logging.getLogger(__name__)


# List of scipy optimizers supported
SCIPY_OPTIMIZERS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",  # requires positive semi definite hessian
    "trust-ncg",
    "trust-exact",  # requires hessian
    "trust-krylov",
]

TORCH_OPTIMIZERS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]


class SolverBase(object):
    """Base class for solver.

    Params:
        image_shape (tuple) ... (H, W)
        calibration_parameter (dict) ... dictionary of the calibration parameter
        solver_config (dict) ... solver configuration
        visualize_module ... visualizer.Visualizer
    """

    def __init__(
        self,
        orig_image_shape: tuple,
        crop_image_shape: tuple,
        calibration_parameter: dict = {},
        solver_config: dict = {},
        visualize_module: Optional[visualizer.Visualizer] = None,
    ):
        self.orig_image_shape = orig_image_shape
        self.crop_image_shape = crop_image_shape
        self.padding = (
            solver_config["outer_padding"] if "outer_padding" in solver_config.keys() else 0
        )
        self.pad_image_shape = (
            crop_image_shape[0] + self.padding,
            crop_image_shape[1] + self.padding,
        )
        self.calib_param = calibration_parameter
        self.slv_config = solver_config
        self.visualizer = visualize_module
        self.setup_filter_preprocess()

        # Cuda utilization
        self._cuda_available = torch.cuda.is_available()
        if self._cuda_available:
            logger.info("Use cuda!")
            self._device = "cuda"
        else:
            self._device = "cpu"

        # For imaging: visualization and CMax (if necessary).
        self.orig_imager = event_image_converter.EventImageConverter(self.orig_image_shape)
        self.crop_imager = event_image_converter.EventImageConverter(self.crop_image_shape, outer_padding=self.padding)  # type: ignore
        # Warp for visualization
        self.normalize_t_in_batch = True  # assume always use displacement, not velocity.
        self.orig_warper = warp.Warp(self.orig_image_shape, normalize_t=self.normalize_t_in_batch, calib_param=self.calib_param)  # type: ignore
        self.crop_warper = warp.Warp(self.crop_image_shape, normalize_t=self.normalize_t_in_batch, calib_param=self.calib_param)  # type: ignore

        self.previous_frame_best_estimation = None
        self.sequential_video_list: List[str] = list()
        self.evaluation_text_list: List[str] = list()
        self.iwe_visualize_max_scale = 50 if not "max_scale" in self.slv_config.keys() else self.slv_config["max_scale"]  # type: ignore
        logger.info(f"Configuration: \n    {self.slv_config}")

    def setup_filter_preprocess(self):
        if "filter" in self.slv_config:
            self.preproc_filter = True
            self.filter_set = utils.EventFilter(self.orig_image_shape, self.slv_config["filter"])
            self.crop_xmin = self.slv_config["filter"]["parameters"]["xmin"]
            self.crop_xmax = self.slv_config["filter"]["parameters"]["xmax"]
            self.crop_ymin = self.slv_config["filter"]["parameters"]["ymin"]
            self.crop_ymax = self.slv_config["filter"]["parameters"]["ymax"]
        else:
            logger.info("No filtering process for events!")
            self.preproc_filter = False
            self.crop_xmin, self.crop_ymin = 0, 0
            self.crop_xmax, self.crop_ymax = self.orig_image_shape

    # Main functions
    def preprocess(self, events: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess events.

        Args:
            events (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        num_orig = len(events)
        time_period = events[:, 2].max() - events[:, 2].min()
        if self.preproc_filter:
            events = self.filter_set.process(events)
            logger.info(f"After preprocessng {len(events)} out of {num_orig}.")

        logger.info(f"Event stats: {len(events)} events, in {time_period} sec.")
        return events, time_period

    def estimate(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Run optimization.

        Inputs:
            events (np.ndarray) ... [n_events x 4] event array. Should be (x, y, t, p).

        Returns:
            (np.ndarray) ... Best optical flow array. [2, H, W].
        """
        raise NotImplementedError

    # Visualizations
    # Helper function
    def create_clipped_image(self, events: np.ndarray, max_scale=50):
        """Creeate IWE for visualization.

        Args:
            events (_type_): _description_
            max_scale (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: np.ndarray
        """
        assert events.shape[-1] <= 4, "this function is for events"
        if isinstance(events, torch.Tensor):
            events = events.clone().detach().cpu().numpy()
        im = self.orig_imager.create_image_from_events_numpy(
            events, method="bilinear_vote", sigma=0
        )
        # print('-=-=-=-', im.max())
        clipped_iwe = 255 - np.clip(max_scale * im, 0, 255).astype(np.uint8)
        if self.padding > 0:
            clipped_iwe = clipped_iwe[self.padding : -self.padding, self.padding : -self.padding]
        return clipped_iwe

    # One-frame visualization
    def visualize_one_batch_warp(self, events: np.ndarray, warp: Optional[np.ndarray] = None):
        if warp is not None:
            if isinstance(warp, torch.Tensor):
                warp = warp.clone().detach().cpu().numpy()
            else:
                warp = np.copy(warp)
            events, _ = self.orig_warper.warp_event(events, warp, self.motion_model, direction='middle')
            flow = self.motion_to_dense_flow(warp)
        clipped_iwe = self.create_clipped_image(events, max_scale=self.iwe_visualize_max_scale)
        self.visualizer.visualize_image(clipped_iwe)
        if warp is not None:
            self.visualizer.visualize_optical_flow_on_event_mask(flow, events)

    def visualize_one_batch_warp_gt(
        self, events: np.ndarray, gt_warp: np.ndarray, motion_model: str = "dense-flow"
    ):
        """
        Args:
            events (np.ndarray): [description]
            gt_warp (np.ndarray): If flow, [H, W, 2]. If other, [motion_dim].
            motion_model (str): motion model, defaults to 'dense-flow'
        """
        if motion_model == "dense-flow":
            gt_warp = np.transpose(gt_warp, (2, 0, 1))  # [2, H, W]
        events, _ = self.orig_warper.warp_event(events, gt_warp, motion_model=motion_model, direction='middle')
        clipped_iwe = self.create_clipped_image(events, max_scale=self.iwe_visualize_max_scale)
        self.visualizer.visualize_image(clipped_iwe)  # type: ignore
        if motion_model == "dense-flow":
            self.visualizer.visualize_overlay_optical_flow_on_event(gt_warp, clipped_iwe)  # type: ignore

    # Sequential visualization
    def visualize_original_sequential(self, orig_events: np.ndarray, filter_events: np.ndarray):
        """Visualize sequential, original image
        Args:
            events (np.ndarray): [description]
            pred_motion (np.ndarray)
        """
        self.visualizer.visualize_event(orig_events, file_prefix="original")  # type: ignore
        if "original" not in self.sequential_video_list:
            self.sequential_video_list.append("original")

        # self.visualizer.visualize_event(orig_events, grayscale=False, file_prefix="original_color")  # type: ignore
        # if "original_color" not in self.sequential_video_list:
        #     self.sequential_video_list.append("original_color")

        clipped_iwe = self.create_clipped_image(
            filter_events, max_scale=self.iwe_visualize_max_scale
        )
        self.visualizer.visualize_image(clipped_iwe, file_prefix="original_filter")  # type: ignore
        if "original_filter" not in self.sequential_video_list:
            self.sequential_video_list.append("original_filter")

    def visualize_pred_sequential(self, events: np.ndarray, flow: np.ndarray):
        """Visualize sequential, prediction
        Args:
            events (np.ndarray): [description]
            flow (np.ndarray)
        """
        self.visualizer.visualize_optical_flow(flow[0], flow[1], visualize_color_wheel=False, file_prefix="pred_flow", save_flow=True)  # type: ignore
        if "pred_flow" not in self.sequential_video_list:
            self.sequential_video_list.append("pred_flow")

        self.visualizer.visualize_poisson_integration(flow, file_prefix="pred_flow_poisson")
        if "pred_flow_poisson" not in self.sequential_video_list:
            self.sequential_video_list.append("pred_flow_poisson")

        # self.visualizer.visualize_overlay_optical_flow_on_event(flow, iwe, file_prefix="pred_overlay")  # type: ignore
        # if "pred_overlay" not in self.sequential_video_list:
        #     self.sequential_video_list.append("pred_overlay")

        self.visualizer.visualize_optical_flow_on_event_mask(flow, events, file_prefix="pred_masked", mask_color="black", mask_morph=True)  # type: ignore
        # TODO CAREFUL! Mask is after filtering.
        if "pred_masked" not in self.sequential_video_list:
            self.sequential_video_list.append("pred_masked")

    def visualize_gt_sequential(self, events: np.ndarray, gt_flow: np.ndarray):
        """Visualize sequential, GT
        Args:
            events (np.ndarray): [description]
            gt_flow (np.ndarray): [2, H, W]
        """
        # Flow
        self.visualizer.visualize_optical_flow(gt_flow[0], gt_flow[1], visualize_color_wheel=False, file_prefix="gt_flow", save_flow=False)  # type: ignore
        if "gt_flow" not in self.sequential_video_list:
            self.sequential_video_list.append("gt_flow")

        self.visualizer.visualize_poisson_integration(gt_flow, file_prefix="gt_flow_poisson")
        if "gt_flow_poisson" not in self.sequential_video_list:
            self.sequential_video_list.append("gt_flow_poisson")

        # self.visualizer.visualize_overlay_optical_flow_on_event(gt_flow, iwe, file_prefix="gt_overlay")  # type: ignore
        # if "gt_overlay" not in self.sequential_video_list:
        #     self.sequential_video_list.append("gt_overlay")

        self.visualizer.visualize_optical_flow_on_event_mask(gt_flow, events, file_prefix="gt_masked", mask_color="black", mask_morph=True)  # type: ignore
        if "gt_masked" not in self.sequential_video_list:
            self.sequential_video_list.append("gt_masked")

    # Optical flow
    def visualize_flows(
        self,
        pred_flow: np.ndarray,
        gt_flow: np.ndarray,
    ) -> None:
        """Visualize the comparison between predicted motion and GT optical flow.

        Args:
            pred_flow (np.ndarray): [2, H, W]. Estimated pixel displacement.
            gt_flow (np.ndarray): [2, H, W]. Pixel displacement.
        """
        self.visualizer.visualize_optical_flow_pred_and_gt(pred_flow, gt_flow, pred_file_prefix="flow_comparison_pred", gt_file_prefix="flow_comparison_gt")  # type: ignore

    def calculate_flow_error(
        self,
        pred_disp: np.ndarray,
        gt_flow: np.ndarray,
        timescale: float = 1.0,
        events: Optional[np.ndarray] = None,
        roi: Optional[dict]=None
    ) -> dict:
        """Calculate optical flow error based on GT.

        Args:
            pred_flow (np.ndarray): [2, H, W] Pixel displacement.
            gt_flow (np.ndarray): [2, H, W]. Pixel displacement.
            timescale (float): timestamp [just for log].
            # TODO question, evaluation on filtered or original??

        Returns:
            dict: flow error dict.
        """
        if events is not None:
            event_mask = self.orig_imager.create_eventmask(events)[:, roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]

        else:
            event_mask = None
        # fwl = {}
        flow_error = utils.calculate_flow_error_numpy(gt_flow[None], pred_disp[None], event_mask=event_mask)  # type: ignore
        # flow_error.update(fwl)
        logger.info(f"{flow_error = } for time period {timescale} sec.")
        return flow_error

    def calculate_fwl(
        self,
        flow: np.ndarray,
        events: np.ndarray,
    ) -> dict:
        """Calculate FWL (from Stoffregen 2020)
        ATTENTION this returns Var(IWE_orig) / Var(IWE) , Less than 1 is better.

        Args:
            flow (np.ndarray): [2, H, W] pixel displacement (not velocity).
            events (np.ndarray): [n, 4]

        Returns:
            dict: flow error dict.
        """
        orig_iwe = self.orig_imager.create_iwe(events)
        warp, _ = self.orig_warper.warp_event(events, flow, "dense-flow", direction='middle')
        iwe = self.orig_imager.create_iwe(warp)
        fwl = costs.NormalizedImageVariance().calculate(
            {"orig_iwe": orig_iwe, "iwe": iwe, "omit_boundary": False}
        )
        return fwl

    def save_flow_error_as_text(
        self, nth_frame: int, flow_error_dict: dict, fname: str = "flow_error_per_frame.txt"
    ):
        if self.visualizer is not None:
            save_file_name = os.path.join(self.visualizer.save_dir, fname)
        else:
            save_file_name = fname
        with open(save_file_name, "a") as f:
            f.write(f"frame {nth_frame}::" + str(flow_error_dict) + "\n")

        if save_file_name not in self.evaluation_text_list and fname != "timestamps_per_frame.txt":
            self.evaluation_text_list.append(save_file_name)

    def set_previous_frame_best_estimation(self, previous_best: np.ndarray):
        if isinstance(previous_best, np.ndarray):
            self.previous_frame_best_estimation = np.copy(previous_best)
        elif isinstance(previous_best, torch.Tensor):
            self.previous_frame_best_estimation = torch.clone(previous_best)
        elif isinstance(previous_best, dict):
            self.previous_frame_best_estimation = previous_best.copy()

    def undistort_image(self, image: np.ndarray):
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.calib_param["K"],
            self.calib_param["D"],
            self.orig_image_shape,
            1,
            self.orig_image_shape,
        )
        undistorted_image = cv2.undistort(
            image,
            self.calib_param["K"],
            self.calib_param["D"],
            None,
            newcameramtx,
        )
        return undistorted_image
