from scipy.ndimage.filters import gaussian_filter
import logging
from typing import Optional, Tuple, Dict
import glob
import os

import cv2
import numpy as np
import optuna
import torch
import torchvision.transforms.functional
from torchvision import transforms

from . import scipy_autograd

from .. import solver, visualizer, utils, costs, types
from .generative_max_likelihood import GenerativeMaximumLikelihood, LossVideosMaker
from ..utils.frame_utils import range_norm

logger = logging.getLogger(__name__)


class PatchEklt(GenerativeMaximumLikelihood):
    def __init__(
            self,
            orig_image_shape: tuple,
            crop_image_shape: tuple,
            calibration_parameter: dict,
            solver_config: dict = {},
            visualize_module: Optional[visualizer.Visualizer] = None,
    ) -> None:
        """Method to determine Optical flow from frames and events inspired by
        https://www.zora.uzh.ch/id/eprint/197701/1/eklt_ijcv19.pdf on patches
        of the frame. Intermediate pixel values are interpolated from the patch
        results

        Args:
            orig_image_shape:
            crop_image_shape:
            calibration_parameter:
            solver_config:
            visualize_module:
        """
        super().__init__(
            orig_image_shape,
            crop_image_shape,
            calibration_parameter,
            solver_config,
            visualize_module,
        )
        self._patch_eklt_config = self.slv_config["patch_eklt"]
        # Let's prepare patch for the whole image first.
        self.patch_size = (self._patch_eklt_config["patch_size"], self._patch_eklt_config["patch_size"])
        if "sliding_window" in self._patch_eklt_config.keys():
            self.sliding_window = (self._patch_eklt_config["sliding_window"], self._patch_eklt_config["sliding_window"])
        else:
            logger.info("Setting sliding window as the patch size..")
            self.sliding_window = self.patch_size
        self.patches, self.patch_image_size = self.prepare_patch(self.orig_image_shape, self.patch_size, self.sliding_window)
        self.n_patch = len(self.patches.keys())
        # Event thresholding
        self.do_event_thresholding = self._patch_eklt_config["do_event_thresholding"]
        self.event_thres = (
            self._patch_eklt_config["event_thres"]
            if "event_thres" in self._patch_eklt_config.keys()
            else None
        )
        self.n_pixel_downsample = 1   # this is to handle low-res data term, only effective in PatchEkltPyramidDynamic

    def prepare_patch(
        self, image_size: tuple, patch_size: tuple, sliding_window: tuple
    ) -> Tuple[Dict[int, types.FlowPatch], tuple]:
        """Get list of patches.

        Args:
            image_size (tuple): (H, W)
            patch_size (tuple): (H, W)
            sliding_window (tuple): (H, W)

        Returns:
            [type]: [description]
        """
        image_h, image_w = image_size
        patch_h, patch_w = patch_size
        slide_h, slide_w = sliding_window
        center_x = np.arange(0, image_h - patch_h + slide_h, slide_h) + patch_h / 2
        center_y = np.arange(0, image_w - patch_w + slide_w, slide_w) + patch_w / 2
        xx, yy = np.meshgrid(center_x, center_y)
        patch_shape = xx.T.shape  # has to be here
        xx, yy = xx.T.reshape(-1), yy.T.reshape(-1)
        patches = {
            i: types.FlowPatch(x=xx[i], y=yy[i], shape=patch_size, u=0.0, v=0.0)
            for i in range(0, len(xx))
        }
        return patches, patch_shape


    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def estimate(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self._gml_config["model_image"] == "current":
            self._set_frame(kwargs["frame"])
        elif self._frame is None and self._gml_config["model_image"] == "background":
            self._set_frame(kwargs["background"])

        # estimate coarse flow array
        patched_flow = np.zeros((2,) + self.patch_image_size, dtype=float).reshape(2, -1)
        self.calculate_iwe_cache(events)

        for i in range(self.n_patch):
            if self.patches[i].x < self.crop_xmin or self.crop_xmax < self.patches[i].x:
                continue
            if self.patches[i].y < self.crop_ymin or self.crop_ymax < self.patches[i].y:
                continue
            
            # Window is inside the whole cropping
            cropped = utils.crop_event(
                events,
                self.patches[i].x_min,
                self.patches[i].x_max,
                self.patches[i].y_min,
                self.patches[i].y_max,
            )

            if not self.do_event_thresholding or len(cropped) > self.event_thres:
                patch_result, _ = self._estimate_patch(events, self.patches[i])
                if self.is_angle_model:
                    patched_flow[:, i] = np.sin(patch_result["angle"]), np.cos(patch_result["angle"])
                else:
                    patched_flow[:, i] = patch_result["v_x"], patch_result["v_y"]

        dense_flow = self.interpolate_dense_flow_from_patch_numpy(patched_flow)
        del self.cache_histogram, self.cache_weights  # free cache
        self.iter_cnt += 1
        return dense_flow

    def interpolate_dense_flow_from_patch_numpy(self, flow_array: np.ndarray) -> np.ndarray:
        """
        Interpolate dense flow from patch.
        Args:
            flow_array (np.ndarray): [2 x h_patch x w_patch]

        Returns:
            np.ndarray: [2 x H x W]
            if self.n_pixel_downsample is not 1, the output shape will be
                (H / self.n_pixel_downsample, W / self.n_pixel_downsample)
        """
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + 1
        flow_array = np.pad(
            flow_array.reshape((2, ) + self.patch_image_size),
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
        )

        interp = cv2.INTER_LINEAR # or cv2.INTER_NEAREST
        scaler_x = self.sliding_window[1] // self.n_pixel_downsample  # attention with coordinate
        scaler_y = self.sliding_window[0] // self.n_pixel_downsample
        upscaled_u = cv2.resize(flow_array[0], None, None, fx=scaler_x, fy=scaler_y, interpolation=interp)
        upscaled_v = cv2.resize(flow_array[1], None, None, fx=scaler_x, fy=scaler_y, interpolation=interp)
        dense_flow = np.concatenate([upscaled_u[None, ...], upscaled_v[None, ...]], axis=0)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        # Obtain center
        output_h = self.orig_image_shape[0] // self.n_pixel_downsample
        output_w = self.orig_image_shape[1] // self.n_pixel_downsample
        h1 = cx - output_h // 2
        w1 = cy - output_w // 2
        h2 = h1 + output_h
        w2 = w1 + output_w
        return dense_flow[..., h1:h2, w1:w2]

    def interpolate_dense_flow_from_patch_tensor(self, flow_array: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_array (np.ndarray): 1-d array, [2 x h_patch x w_patch]

        Returns:
            np.ndarray: [2 x H x W]
            if self.n_pixel_downsample is not 1, the output shape will be
                (H / self.n_pixel_downsample, W / self.n_pixel_downsample)
        """
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + 1
        flow_array = torch.nn.functional.pad(
            flow_array.reshape((1, 2) + self.patch_image_size),
            (pad_w, pad_w, pad_h, pad_h),
            mode="replicate",
        )[0]
        interp = transforms.InterpolationMode.BILINEAR
        size = [
            flow_array.shape[1] * self.sliding_window[0] // self.n_pixel_downsample,
            flow_array.shape[2] * self.sliding_window[1] // self.n_pixel_downsample,
        ]
        dense_flow = transforms.functional.resize(flow_array, size, interpolation=interp)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        # Obtain center
        output_h = self.orig_image_shape[0] // self.n_pixel_downsample
        output_w = self.orig_image_shape[1] // self.n_pixel_downsample
        h1 = cx - output_h // 2
        w1 = cy - output_w // 2
        h2 = h1 + output_h
        w2 = w1 + output_w
        return dense_flow[..., h1:h2, w1:w2]

    def interpolate_dense_poisson_from_patch_numpy(self, intensity_array: np.ndarray) -> np.ndarray:
        """
        Interpolate intensity flow from patch.
        Args:
            intensity_array (np.ndarray): [h_patch x w_patch]

        Returns:
            np.ndarray: [H x W]
            if self.n_pixel_downsample is not 1, the output shape will be
                (H / self.n_pixel_downsample, W / self.n_pixel_downsample)
        """
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + 1
        intensity_array = np.pad(
            intensity_array.reshape((1, ) + self.patch_image_size),
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="edge",
        )
        interp = cv2.INTER_LINEAR # or cv2.INTER_NEAREST
        scaler_x = self.sliding_window[1] // self.n_pixel_downsample  # attention with coordinate
        scaler_y = self.sliding_window[0] // self.n_pixel_downsample
        upscaled = cv2.resize(intensity_array[0], None, None, fx=scaler_x, fy=scaler_y, interpolation=interp)
        cx, cy = upscaled.shape[0] // 2, upscaled.shape[1] // 2
        # Obtain center
        output_h = self.orig_image_shape[0] // self.n_pixel_downsample
        output_w = self.orig_image_shape[1] // self.n_pixel_downsample
        h1 = cx - output_h // 2
        w1 = cy - output_w // 2
        h2 = h1 + output_h
        w2 = w1 + output_w
        return upscaled[h1:h2, w1:w2]

    def interpolate_dense_poisson_from_patch_tensor(self, intensity_array: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intensity_array (np.ndarray): [h_patch x w_patch]

        Returns: [H x W]
            if self.n_pixel_downsample is not 1, the output shape will be
                (H / self.n_pixel_downsample, W / self.n_pixel_downsample)
        """
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + 1
        intensity_array = torch.nn.functional.pad(
            intensity_array.reshape((1, 1) + self.patch_image_size),
            (pad_w, pad_w, pad_h, pad_h),
            mode="replicate",
        )[0]
        interp = transforms.InterpolationMode.BILINEAR
        size = [
            intensity_array.shape[1] * self.sliding_window[0] // self.n_pixel_downsample,
            intensity_array.shape[2] * self.sliding_window[1] // self.n_pixel_downsample,
        ]
        dense = transforms.functional.resize(intensity_array, size, interpolation=interp)
        cx, cy = dense.shape[1] // 2, dense.shape[2] // 2
        # Obtain center
        output_h = self.orig_image_shape[0] // self.n_pixel_downsample
        output_w = self.orig_image_shape[1] // self.n_pixel_downsample
        h1 = cx - output_h // 2
        w1 = cy - output_w // 2
        h2 = h1 + output_h
        w2 = w1 + output_w
        return dense[0, h1:h2, w1:w2]


    def calculate_iwe_cache(self, events):
        """Calculate cache of IWE for speed up.

        Args:
            events (_type_): _description_
        """
        pol_image = self.orig_imager.create_iwe(events, method="polarity", sigma=0)
        if self._gml_config["no_polarity"]:
            histogram = pol_image[0] + pol_image[1]
        else:
            histogram = pol_image[0] - pol_image[1]  # positive - negative

        if self._gml_config["weight_loss_by_event_hist"]:
            self.cache_weights = cv2.GaussianBlur(np.abs(histogram), ksize=None,
                                       sigmaX=self._gml_config["weight_sigma"])
        else:
            self.cache_weights = None

        if self._gml_config["iwe_sigma"]:
            self.cache_histogram = cv2.GaussianBlur(histogram, ksize=None,
                                         sigmaX=self._gml_config["iwe_sigma"])
        else:
            self.cache_histogram = histogram

        if self.do_weight_inverse:
            self.weight_inverse = gaussian_filter(np.abs(histogram), 10)
            # print('-=-=--', self.weight_inverse.max(), self.weight_inverse.mean(), self.weight_inverse.min())
            self.weight_inverse = np.clip(self.weight_inverse, 0, self.weight_inverse.mean() + self.weight_inverse.std() / 2.)
            # print('-=-=--', self.weight_inverse.max(), self.weight_inverse.mean(), self.weight_inverse.min())
            # raise RuntimeError
            self.weight_inverse /= self.weight_inverse.max()
            self.weight_inverse = 1.0 - 0.95 * self.weight_inverse
            # self.weight_inverse = 1.0 / (self.weight_inverse + 1e-2)
        else:
            self.weight_inverse = np.ones_like(histogram)
        # self.weight_inverse = torch.from_numpy(self.weight_inverse).double().to(self._device)


    def _make_measured_increment(self, events: np.ndarray, roi: dict) -> np.ndarray:
        """Overload: optimized version of GenerativeMaximumLikelihood._make_measured_increment function.
        """
        x_min, x_max = roi["xmin"], roi["xmax"]
        y_min, y_max = roi["ymin"], roi["ymax"]
        measured_increment = self.cache_histogram[x_min: x_max, y_min: y_max]

        if self.cache_weights is not None:
            weights = self.cache_weights[x_min: x_max, y_min: y_max]
            measured_increment = weights * measured_increment
        else:
            weights = None
        measured_increment /= np.linalg.norm(measured_increment)
        if logger.isEnabledFor(logging.DEBUG):
            self.visualizer.visualize_image(range_norm(measured_increment, dtype=np.uint8),
                                            file_prefix="hist")
        return measured_increment, weights


    def _make_prediction_torch(self, parameters: torch.Tensor, roi: dict, weights: torch.Tensor):
        """Overload: optimized version of GenerativeMaximumLikelihood._make_prediction_torch
        """
        assert self.is_angle_model
        v_x, v_y = torch.sin(parameters[0]), torch.cos(parameters[0])
        x_min, x_max = roi["xmin"], roi["xmax"]
        y_min, y_max = roi["ymin"], roi["ymax"]

        gradient_x = self._gradient_x_torch.clone()[x_min: x_max, y_min: y_max]
        gradient_y = self._gradient_y_torch.clone()[x_min: x_max, y_min: y_max]

        if self._gml_config["optimize_warp"]:
            p_x, p_y = parameters[1], parameters[2]
            translation = torch.Tensor([p_x, p_y])
            gradient_x = utils.frame_utils.warp_image_torch(gradient_x, translation)
            gradient_y = utils.frame_utils.warp_image_torch(gradient_y, translation)

        predicted_increment = v_x * gradient_x + v_y * gradient_y
        if self._gml_config["no_polarity"]:
            predicted_increment = np.abs(predicted_increment)

        if weights is not None:
            predicted_increment *= weights
        predicted_increment /= torch.linalg.norm(predicted_increment.clone()) + 0.0001
        return predicted_increment

