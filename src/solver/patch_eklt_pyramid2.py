import logging
from typing import Optional, Tuple, Dict
import math
import cv2
import numpy as np
import torch
import os

from torchvision.transforms.functional import gaussian_blur, resize
from . import scipy_autograd
import skimage

from .. import solver, visualizer, utils, costs, types
from .generative_max_likelihood import LossVideosMaker
from ..utils.frame_utils import range_norm
from .patch_eklt_dependent import PatchEkltDependent


logger = logging.getLogger(__name__)


class PatchEkltPyramid2(PatchEkltDependent):
    def __init__(
            self,
            orig_image_shape: tuple,
            crop_image_shape: tuple,
            calibration_parameter: dict,
            solver_config: dict = {},
            visualize_module: Optional[visualizer.Visualizer] = None,
    ) -> None:
        """Pyramidal (coarse-to-fine) estimation

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
        # self.prepare_pyramidal_patch((768, 1280), 32, 4, offset=(24, 0))
        # self.prepare_pyramidal_patch(self.orig_image_shape, 64, 8)
        self.prepare_pyramidal_patch(self.orig_image_shape, 64, 8)
        self.overload_patch_configuration(self.coarest_scale)
        self.estimate_mask_dense_numpy = np.zeros(self.orig_image_shape)
        self.estimate_mask_dense_numpy[self.crop_xmin:self.crop_xmax, self.crop_ymin:self.crop_ymax] = 1

    # Bootstrap functions
    def prepare_pyramidal_patch(self, image_size: tuple, coarsest_patch_size: int, finest_patch_size: int, offset: tuple=(0, 0)):
        """To achieve pyramidal patch, set special member variables.
        You can use `overload_patch_configuration` to set the current scale.

        Args:
            image_size (tuple): [description]
            scales (int): [description]
        """
        self.coarest_scale = 1
        self.finest_scale = int(np.log2(coarsest_patch_size / finest_patch_size)) + 2
        logger.info(f'{self.coarest_scale = }, {self.finest_scale = }')
        self.scaled_patches = {}
        self.scaled_patch_image_size = {}
        self.scaled_n_patch = {}
        self.scaled_patch_size = {}
        self.scaled_sliding_window = {}
        self.total_n_patch = 0
        self.current_scale = self.coarest_scale
        self.scaled_imager = {}
        self.scaled_warper = {}
        for i in range(self.coarest_scale, self.finest_scale):
            scaled_size = (coarsest_patch_size // (2**(i - 1)), coarsest_patch_size // (2**(i - 1)))
            self.scaled_patch_size[i] = scaled_size
            self.scaled_sliding_window[i] = scaled_size
            self.scaled_patches[i], self.scaled_patch_image_size[i] = self.prepare_patch(
                image_size, scaled_size, scaled_size, offset
            )
            self.scaled_n_patch[i] = len(self.scaled_patches[i].keys())
            self.total_n_patch += self.scaled_n_patch[i]

    def prepare_patch(
        self, image_size: tuple, patch_size: tuple, sliding_window: tuple, offset: tuple=(0, 0)
    ) -> Tuple[Dict[int, types.FlowPatch], tuple]:
        """Get list of patches.

        Args:
            image_size (tuple): (H, W)
            patch_size (tuple): (H, W)
            sliding_window (tuple): (H, W)
            offset (tuple): (H, W) Where the actual image starts.

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
            i: types.FlowPatch(x=xx[i] - offset[0], y=yy[i] - offset[1], shape=patch_size, u=0.0, v=0.0)
            for i in range(0, len(xx))
        }
        return patches, patch_shape


    def overload_patch_configuration(self, n_scale: int):
        """Overload the related member variables set to the current scale.

        Args:
            n_scale (int): 0 is original size. 1 is half size, etc.
        """
        self.current_scale = n_scale
        self.patches = self.scaled_patches[n_scale]
        self.patch_image_size = self.scaled_patch_image_size[n_scale]
        self.patches = self.scaled_patches[n_scale]
        self.n_patch = self.scaled_n_patch[n_scale]
        self.sliding_window = self.scaled_sliding_window[n_scale]
        self.sliding_window_h = self.sliding_window[0]
        self.sliding_window_w = self.sliding_window[1]
        self.patch_size = self.scaled_patch_size[n_scale]


    @utils.profile(
        output_file="optimize.prof", sort_by="cumulative", lines_to_print=300, strip_dirs=True
    )
    def estimate(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self._gml_config["model_image"] == "current":
            self._set_frame(kwargs["frame"])
        elif self._gml_config["model_image"] == "black":
            self._set_frame(np.zeros_like(kwargs["frame"]))
        elif self._frame is None and self._gml_config["model_image"] == "background":
            self._set_frame(kwargs["background"])

        # estimate coarse flow array
        self.calculate_iwe_cache(events)

        best_params_per_scale = {}
        # Coarse to fine
        for s in range(self.coarest_scale, self.finest_scale):
            self.overload_patch_configuration(s)
            logger.info(f"Scale {self.current_scale}, patch num {self.patch_image_size}, patch shape {self.patch_size}")
            opt_result = self.run_estimation_per_scale(events, best_params_per_scale)
            best_params_per_scale[s] = opt_result.reshape(
                ((-1,) + self.patch_image_size)
            )

        if self.is_poisson_model:
            patch_flow = self.poisson_to_flow(best_params_per_scale[self.current_scale][0])
            dense_flow = self.interpolate_dense_flow_from_patch_numpy(patch_flow)
            # dense_poisson = self.interpolate_dense_poisson_from_patch_numpy(best_params_per_scale[self.current_scale][0])
            # dense_flow = self.poisson_to_flow(dense_poisson)
        else:
            dense_flow = self.interpolate_dense_flow_from_patch_numpy(best_params_per_scale[self.current_scale][:2])
        self.visualizer.visualize_scipy_history(self.cost_func.get_history())
        self._video_maker.make_video()
        self.cost_func.clear_history()

        if self._gml_config["optimize_warp"]:
            translation = self.interpolate_dense_flow_from_patch_numpy(best_params_per_scale[self.current_scale][-2:])
            roi_translation = translation[:, self.crop_xmin:self.crop_xmax, self.crop_ymin:self.crop_ymax]
            logger.debug(f"""p_x, p_y ... Max: {roi_translation.max()}, Min: {roi_translation.min()}
            Norm(x): {np.abs(roi_translation[0]).mean()}
            Norm(y): {np.abs(roi_translation[1]).mean()}""")

        # # Scipy optimizers
        # result = scipy_autograd.minimize(
        #     lambda x: self._objective_scipy(x, measured_increment, roi, weights),
        #     x0=x0,
        #     method=self._opt_method,
        #     options={"gtol": 1e-8, "disp": True},
        #     precision='float64',
        # )
        # if not result.success:
        #     logger.warning("Unsuccessful optimization step!")
        # dense_flow = self._extrapolate_dense_flow_from_estimates(result.x)
        del self.cache_histogram, self.cache_weights  # free cache
        self.iter_cnt += 1

        # Have the current estimation result for the next frame initialization
        # best_motion_per_scale_feedback = self.update_coarse_from_fine(best_params_per_scale)
        # self.set_previous_frame_best_estimation(best_motion_per_scale_feedback)
        # self.set_previous_frame_best_estimation(best_params_per_scale)
        return dense_flow * self.estimate_mask_dense_numpy

    def get_patch_pad_shape(self):
        center_x = np.arange(0, self.orig_image_shape[0] - self.patch_size[0] + self.sliding_window_h, self.sliding_window_h) + self.patch_size[0] / 2
        center_y = np.arange(0, self.orig_image_shape[1] - self.patch_size[1] + self.sliding_window_w, self.sliding_window_w) + self.patch_size[1] / 2
        pad_x0 = len(np.where(center_x < self.crop_xmin)[0])
        pad_x1 = len(np.where(self.crop_xmax < center_x)[0])
        pad_y0 = len(np.where(center_y < self.crop_ymin)[0])
        pad_y1 = len(np.where(self.crop_ymax < center_y)[0])
        return (pad_x0, pad_x1, pad_y0, pad_y1)

    def get_patch_shape(self):
        center_x = np.arange(0, self.crop_image_shape[0] - self.patch_size[0] + self.sliding_window_h, self.sliding_window_h)
        center_y = np.arange(0, self.crop_image_shape[1] - self.patch_size[1] + self.sliding_window_w, self.sliding_window_w)
        return (len(center_x), len(center_y))

    def estimate_mask_dense(self):
        # print(np.where(self.estimate_mask_dense_numpy > 0), self.crop_xmin, self.crop_xmax, self.crop_ymin, self.crop_ymax)
        return torch.from_numpy(np.copy(self.estimate_mask_dense_numpy)).double().to(self._device)

    def run_estimation_per_scale(self, events, param_per_scale):
        # Setup initial paeameters
        # First, get the number of the parameters to estimate, by ROI and threshloding events.
        self.estimate_mask_patch = np.ones(self.patch_image_size)
        for ii in range(self.estimate_mask_patch.shape[0]):
            for jj in range(self.estimate_mask_patch.shape[1]):
                # print('-=-=-=', ii, jj, ii * self.estimate_mask_patch.shape[0] + jj)
                _patch = self.patches[ii * self.estimate_mask_patch.shape[0] + jj]
                if _patch.x  < self.crop_xmin or self.crop_xmax < _patch.x:
                    self.estimate_mask_patch[ii, jj] = 0
                if _patch.y  < self.crop_ymin or self.crop_ymax < _patch.y:
                    self.estimate_mask_patch[ii, jj] = 0
                # Window is inside the whole cropping
                cropped = utils.crop_event(
                    events, _patch.x_min, _patch.x_max, _patch.y_min, _patch.y_max,
                )
                if not self.do_event_thresholding or len(cropped) < self.event_thres:                
                    self.estimate_mask_patch[ii, jj] = 0
                # For now, disable event thresholding
        self.estimate_mask_patch = torch.from_numpy(self.estimate_mask_patch).double().to(self._device)
        self.n_parameter_dim = len(self._initialize_velocity())

        # x0: parameter is always shape (n_dim, h, w) - full patch-image shape.
        if self.previous_frame_best_estimation is not None:
            logger.info("Use previous best motion!")
            if self.current_scale == self.coarest_scale:
                x0 = np.copy(self.previous_frame_best_estimation[self.current_scale])
            elif self.current_scale > self.coarest_scale:
                logger.info("Use the coarser motion!  -  not utilizing the previous frame for fine resolutions.")
                _tx0 = torch.from_numpy(param_per_scale[self.current_scale - 1])
                x0 = resize(_tx0, self.patch_image_size).numpy()
                x0 = (self.previous_frame_best_estimation[self.current_scale] + x0) / 2.
        else:
            if self.current_scale == self.coarest_scale:
                logger.info("Initialize with zero")
                x0 = np.concatenate([self._initialize_velocity() for _ in range(self.n_patch)]).reshape((self.n_parameter_dim, ) + self.patch_image_size)
            elif self.current_scale > self.coarest_scale:
                logger.info("Use the coarser motion!")
                _tx0 = torch.from_numpy(param_per_scale[self.current_scale - 1])
                x0 = resize(_tx0, self.patch_image_size).numpy()
        # print('-0=-=-=-=-', x0.shape)

        x0 = torch.from_numpy(x0).double().to(self._device).requires_grad_()   # leaf tensor
        roi = {"xmin": self.crop_xmin, "xmax": self.crop_xmax, "ymin": self.crop_ymin, "ymax": self.crop_ymax}
        measured_increment_numpy, weights_numpy = self._make_measured_increment(events, roi)
        # measured_increment = torch.from_numpy(measured_increment).double().to(self._device).requires_grad_() * self.estimate_mask_dense()
        # weights = torch.from_numpy(weights).double().to(self._device) * self.estimate_mask_dense() if weights is not None else weights

        # torch optimizers
        lr_step = iters = self._opt_config["n_iter"] // (self.finest_scale - self.current_scale + 1)
        # lr, lr_decay = 0.05, 0.1
        lr, lr_decay = 0.05, 0.1
        # optimizer = torch.optim.__dict__[self._opt_method]([x0], lr=lr)
        optimizer = torch.optim.__dict__[self._opt_method]([x0], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
        best_x, best_it, min_loss = x0, 0, math.inf
        for it in range(iters):
            measured_increment = torch.from_numpy(measured_increment_numpy).double().to(self._device).requires_grad_() * self.estimate_mask_dense()
            weights = torch.from_numpy(weights_numpy).double().to(self._device) * self.estimate_mask_dense() if weights_numpy is not None else None
            optimizer.zero_grad()
            loss = self._objective_scipy(x0, measured_increment, roi, weights)
            
            # visualize evolution     
            self.visualize_evolution(x0, measured_increment, weights, roi)
            if loss < min_loss:
                best_x = x0
                min_loss = loss.item()
                best_it = it
            try:
                loss.backward()
            except Exception as e:
                logger.error(e)
                break
            optimizer.step()
            scheduler.step()

        best_x = best_x.detach().cpu().numpy()
        return best_x


    def _get_patch_flow(self, parameters: torch.Tensor):
        """From parameters that has reduced number of patches, recovers the original patch size flow.
        Args:
            parameters ... 1-dimensional, n_dim x n_target_patches
        """
        if self.is_poisson_model:
            assert self.n_parameter_dim in [1, 3]
            patch_flow = self.poisson_to_flow(parameters[[0]])
        else:
            assert not self.is_angle_model
            assert self.n_parameter_dim in [2, 4]
            patch_flow = parameters[[0, 1]]
        # else:
        #     if self.is_angle_model:
        #         assert self.n_parameter_dim in [1, 3]
        #         v_x, v_y = torch.sin(reshaped_params[0]), torch.cos(reshaped_params[0])   # each has n-patch length
        #     else:
        #         assert self.n_parameter_dim in [2, 4]
        #         v_x, v_y = reshaped_params[0], reshaped_params[1]   # each has n-patch length
        #     patch_flow = patch_flow.reshape((2, ) + self.patch_image_size)
        return patch_flow


    def _get_patch_translation(self, parameters: torch.Tensor):
        """From parameters that has reduced number of patches, recovers the original patch size translation (px, py).
        """
        assert not self.is_angle_model
        return parameters[[-2, -1]]


    def _get_patch_poisson(self, parameters: torch.Tensor):
        """From parameters that has reduced number of patches, recovers the original patch size Poisson (intensity).
        Returns 1, patch_h, patch_w
        """
        assert self.is_poisson_model
        assert self.n_parameter_dim in [1, 3]
        return parameters[[0]]

    def _make_measured_increment(self, events: np.ndarray, roi: dict) -> np.ndarray:
        """Overload: optimized version of GenerativeMaximumLikelihood._make_measured_increment function.
        """
        measured_increment = self.cache_histogram

        if self.cache_weights is not None:
            weights = self.cache_weights
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
        gradient_x = self._gradient_x_torch.clone()
        gradient_y = self._gradient_y_torch.clone()

        dense_flow = self._extrapolate_dense_flow_from_estimates(parameters)

        if self._gml_config["optimize_warp"]:
            translation = self._extrapolate_dense_translation_from_estimates(parameters)
            gradient_x = utils.frame_utils.warp_image_forward(gradient_x, translation)
            gradient_y = utils.frame_utils.warp_image_forward(gradient_y, translation)

        predicted_increment = dense_flow[0] * gradient_x + dense_flow[1] * gradient_y
        if self._gml_config["no_polarity"]:
            predicted_increment = torch.abs(predicted_increment)

        if weights is not None:
            predicted_increment *= weights
        predicted_increment /= torch.linalg.norm(predicted_increment.clone()) + 0.0001
        return predicted_increment * self.estimate_mask_dense()


    def _objective_scipy(self, parameters: torch.Tensor, measured_increment: torch.Tensor,
                         roi: dict, weights: torch.Tensor = None):
        """

        Args:
            parameters:
            measured_increment:
            weights:

        Returns:

        """
        predicted_increment = self._make_prediction_torch(parameters, roi, weights)
        roi_dense_flow = self._extrapolate_dense_flow_from_estimates(parameters) * self.estimate_mask_dense()
        weight_inverse = self.weight_inverse
        cost_kwarg = {"flow": roi_dense_flow,
                      "weights": torch.from_numpy(weight_inverse).double().to(self._device)}
        if self.do_weight_inverse:
            self._video_maker.visualize_image((weight_inverse*255).astype(np.uint8), "opt_mask")
        if self._gml_config["optimize_warp"]:
            roi_translation = self._extrapolate_dense_translation_from_estimates(parameters) * self.estimate_mask_dense()
            # roi_translation = translation[:, roi["xmin"]: roi["xmax"], roi["ymin"]: roi["ymax"]]
            cost_kwarg.update({"pxy": roi_translation})
        if self.is_poisson_model:
            intensity = self._extrapolate_dense_poisson_from_estimates(parameters) * self.estimate_mask_dense()
            cost_kwarg.update({"intensity": intensity})

        cost = self._calculate_cost(measured_increment, predicted_increment, **cost_kwarg)
        logger.debug(f"loss: {cost:.6f}")
        return cost

    def visualize_evolution(self, params, measurement, weights, roi):
        if not logger.isEnabledFor(logging.DEBUG):
            return

        if weights is not None:
            weights = weights.clone().detach()
        prediction = self._make_prediction_torch(params, roi, weights).detach().cpu().numpy()

        dense_flow = self._extrapolate_dense_flow_from_estimates(params).clone().detach().cpu().numpy()
        roi_dense_flow = dense_flow * self.estimate_mask_dense_numpy
        self._video_maker.visualize_flow(roi_dense_flow, "opt_flow")

        if self._gml_config["optimize_warp"]:
            translation = self._extrapolate_dense_translation_from_estimates(params).clone().detach().cpu().numpy()
            roi_translation = translation* self.estimate_mask_dense_numpy
            self._video_maker.visualize_flow(roi_translation, "opt_pxy")
        if self.is_poisson_model:
            # poisson = self._get_patch_poisson(params).clone().detach().cpu().numpy()
            dense_poisson = self._extrapolate_dense_poisson_from_estimates(params).clone().detach().cpu().numpy()
            roi_poisson = dense_poisson * self.estimate_mask_dense_numpy
            self._video_maker.visualize_image(range_norm(roi_poisson, dtype=np.uint8),
                                              "opt_poisson")

        measured_increment = measurement.clone().detach().cpu().numpy()
        diff = prediction - measured_increment
        lower, upper = self._gml_config["viz_diff_scale"]
        d_min, d_max = np.min(diff), np.max(diff)
        if d_min < lower:
            logger.warning(f"The lowest value in diff is {d_min} but lower scale is {lower}")
        if d_max > upper:
            logger.warning(f"The highest value in diff is {d_max} but lower scale is {upper}")
        diff = range_norm(diff, lower=lower,
                        upper=upper, dtype=np.uint8)
        self._video_maker.visualize_image(diff, "opt_diff")
        self._video_maker.visualize_image(range_norm(prediction, dtype=np.uint8),
                                        "opt_prediction")
        self._video_maker.visualize_image(range_norm(measured_increment, dtype=np.uint8),
                                        "opt_measured")


    def update_coarse_from_fine(self, params_per_scale: dict) -> dict:
        """Take average of finer motion and give it feedback toward coarser dimension.
        Args:
            params_per_scale (dict): [description]

        Returns:
            [dict]: [It has always full-size patch.
        """
        refined_motion = {self.finest_scale - 1: params_per_scale[self.finest_scale - 1]}
        for i in range(self.coarest_scale + 1, self.finest_scale):
            self.overload_patch_configuration(i)
            # refined_motion[i - 1] = skimage.transform.pyramid_reduce(
            #     params_per_scale[i].reshape((-1,) + self.patch_image_size), channel_axis=0
            # )
            _tx0 = torch.from_numpy(params_per_scale[i])
            refined_motion[i - 1] = resize(_tx0, self.scaled_patch_image_size[i - 1]).numpy()

        #     print('aaaa', params_per_scale[i].shape, refined_motion[i - 1].shape)
        # raise RuntimeError
        return refined_motion
