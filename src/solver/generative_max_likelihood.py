import logging
from typing import Optional
import glob
import os
import copy
import math

import cv2
import numpy as np
import optuna
import torch
import shutil

from . import scipy_autograd
from .base import SCIPY_OPTIMIZERS, TORCH_OPTIMIZERS

from .. import solver, visualizer, utils, costs
from ..utils.frame_utils import range_norm

logger = logging.getLogger(__name__)

AVAILABLE_MODEL_IMAGES = ["background", "current", "black", "e2vid"]


def run_optuna_config_checks(solver_config, is_angle_mode, pxpy_as_anglemagn):
    params = set(solver_config["optimizer"]["parameters"].keys())
    if is_angle_mode:
        assert {"angle"}.issubset(params)
    else:
        assert {"v_x", "v_y"}.issubset(params)

    if solver_config["generative_ml"]["optimize_warp"]:
        if pxpy_as_anglemagn:
            assert {"p_angle", "p_magn"}.issubset(params)
        else:
            assert {"p_x", "p_y"}.issubset(params)


class LossVideosMaker:
    def __init__(self, image_shape, save_dir, name, log_level=logging.DEBUG):
        self.visualizer = visualizer.Visualizer(image_shape,
                                                save_dir=os.path.join(save_dir, "tmp"),
                                                save=True, show=False)
        self.destination_dir = save_dir
        self.name = name
        self.image_names = []
        self.count = 0
        self.log_level = log_level

    def make_video(self):
        if logger.isEnabledFor(self.log_level):
            for v in self.image_names:
                self.visualizer.visualize_sequential_images_as_video(v)
                for png_file in glob.glob(os.path.join(self.visualizer.save_dir, f'{v}*.png')):
                    if png_file.endswith('.png'):
                        os.remove(png_file)

            if len(self.image_names) > 1:
                out_video = self.visualizer.concat_videos(self.image_names, f"{self.name}{self.count}")
                shutil.move(out_video, os.path.join(self.destination_dir, f"{self.name}{self.count}.mp4"))

            for v in self.image_names:
                mp4_file = os.path.join(self.visualizer.save_dir, f'{v}.mp4')
                os.remove(mp4_file)

            self.visualizer.reset_save_count("all")
            self.image_names = []
            self.count += 1

    def visualize_image(self, image, file_prefix):
        if logger.isEnabledFor(self.log_level):
            file_prefix = f"LVM_{file_prefix}"  # workaround to assure somewhat names are not accidentally double used
            if file_prefix not in self.image_names:
                self.image_names.append(file_prefix)
            self.visualizer.visualize_image(image, file_prefix)

    def visualize_flow(self, flow, file_prefix):
        if logger.isEnabledFor(self.log_level):
            file_prefix = f"LVM_{file_prefix}"  # workaround to assure somewhat names are not accidentally double used
            if file_prefix not in self.image_names:
                self.image_names.append(file_prefix)
            self.visualizer.visualize_optical_flow(flow[0], flow[1], file_prefix=file_prefix)


class GenerativeMaximumLikelihood(solver.SolverBase):
    def __init__(
            self,
            orig_image_shape: tuple,
            crop_image_shape: tuple,
            calibration_parameter: dict,
            solver_config: dict = {},
            visualize_module: Optional[visualizer.Visualizer] = None,
    ) -> None:
        """Method to determine Optical flow from frames and events inspired by
        https://www.zora.uzh.ch/id/eprint/197701/1/eklt_ijcv19.pdf

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
        self._frame = None
        self._gradient_x = None
        self._gradient_y = None
        self._opt_config = self.slv_config["optimizer"]
        self._opt_method = self._opt_config["method"]
        # self._roi = {key: solver_config["filter"]["parameters"][key] for key in ["xmin", "xmax", "ymin", "ymax"]}
        self._gml_config = self.slv_config["generative_ml"]
        self._opt_parameters = self._opt_config["parameters"].keys()
        self._video_maker = LossVideosMaker(orig_image_shape, self.visualizer.save_dir, "optimization")
        self.setup_cost_func()
        self.iter_cnt = 0
        assert self._gml_config["model_image"] in AVAILABLE_MODEL_IMAGES, \
            f"the setting 'mode_image' must be in {AVAILABLE_MODEL_IMAGES}."
        self.is_angle_model = utils.check_key_and_bool(self._gml_config, "angle_model")
        self.is_poisson_model = utils.check_key_and_bool(self._gml_config, "poisson_model")
        self.do_weight_inverse = utils.check_key_and_bool(self._gml_config, "weight_loss_by_inverse_event_hist")
        self.weight_inverse = np.ones(self.orig_image_shape)

        self.pxpy_as_anglemagn = utils.check_key_and_bool(self._gml_config, "px-py_as-angle-magnitude")
        if self._opt_method == "optuna":
            run_optuna_config_checks(self.slv_config, self.is_angle_model, self.pxpy_as_anglemagn)
        else:
            if self.pxpy_as_anglemagn:
                raise NotImplementedError("Currently evaluating px and py as angle and magnitude"
                                          "is only supported for the optuna optimizer")

        self.sobel_ksize = (
            self._gml_config["sobel_ksize"]
            if "sobel_ksize" in self._gml_config.keys()
            else 3
        )
        self.sobel_padding = 2 if self.sobel_ksize == 5 else 1

    def unfold_params(self, params: dict):
        expanded = {}
        if self.is_angle_model:
            expanded["v_x"] = np.sin(params["angle"])
            expanded["v_y"] = np.cos(params["angle"])
        else:
            expanded["v_x"] = params["v_x"]
            expanded["v_y"] = params["v_y"]

        if self._gml_config["optimize_warp"]:
            if self.pxpy_as_anglemagn:
                expanded["p_x"] = params["p_magn"] * np.sin(params["p_angle"])
                expanded["p_y"] = params["p_magn"] * np.cos(params["p_angle"])
            else:
                expanded["p_x"] = params["p_x"]
                expanded["p_y"] = params["p_y"]

        return expanded

    def unfold_scipy_params(self, params):
        if self.is_angle_model:
            converted = {"angle": params[0]}
            params = params[1:]
        else:
            converted = {"v_x": params[0],
                         "v_y": params[1]}
            params = params[2:]

        if self._gml_config["optimize_warp"]:
            if self.pxpy_as_anglemagn:
                converted["p_magn"] = params[0]
                converted["p_angle"] = params[1]
            else:
                converted["p_x"] = params[0]
                converted["p_y"] = params[1]

        return self.unfold_params(params)

    def setup_cost_func(self):
        precision = "64"
        logger.info(f"Load hybrid cost")
        self.cost_weight = self.slv_config["cost_with_weight"]
        self.cost_func = costs.HybridCost(
            direction="minimize",
            cost_with_weight=self.cost_weight,
            store_history=True,
            precision=precision,
            cuda_available=self._cuda_available
        )

    def _set_frame(self, frame: np.ndarray) -> None:
        """Sets the frame image and calculated the gradient maps

        Args:
            frame: the background image of original shape

        Returns:

        """
        logger.info("Setting new background image for flow estimation.")
        if self._gml_config["use_log_intensity"]:
            frame = np.log(frame + 1).astype(float)
        self._frame = frame
        self._gradient_x = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        self._gradient_y = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)

        if self._opt_method in SCIPY_OPTIMIZERS or self._opt_method in TORCH_OPTIMIZERS:
            # Convert to pytorch
            self._gradient_x_torch = torch.from_numpy(self._gradient_x).to(self._device).double()
            self._gradient_y_torch = torch.from_numpy(self._gradient_y).to(self._device).double()

    def _run_optuna(self, measured_increment: np.ndarray, roi: dict, weights: np.ndarray = None):
        if self._opt_config["sampler"] == "TPE":
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=max(10, self._opt_config["n_iter"] // 10)
            )
        elif self._opt_config["sampler"] == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self._opt_config["sampler"] in ["grid", "uniform"]:
            sampler = self._make_uniform_sampler()
        else:
            e = f"Sampling method {self._opt_config['sampler']} is not supported"
            logger.error(e)
            raise NotImplementedError(e)

        study = optuna.create_study(
            direction="minimize", sampler=sampler, storage=utils.SingleThreadInMemoryStorage()
        )
        study.optimize(
            lambda trial: self._objective_optuna(trial, measured_increment, roi, weights),
            n_trials=self._opt_config["n_iter"]
        )
        return study

    def _make_uniform_sampler(self):
        min_max = {
            k: [
                self._opt_config["parameters"][k]["min"],
                self._opt_config["parameters"][k]["max"],
            ]
            for k in self._opt_parameters
        }
        search_space = {
            k: np.arange(
                min_max[k][0],
                min_max[k][1],
                (min_max[k][1] - min_max[k][0]) / self._opt_config["n_iter"],  # type: ignore
            )
            for k in self._opt_parameters
        }
        sampler = optuna.samplers.GridSampler(search_space)
        return sampler

    def _objective_optuna(self, trial: optuna.Trial, measured_increment: np.ndarray,
                          roi: dict, weights: np.ndarray = None):
        sample = {k: self._sampling(trial, k) for k in self._opt_parameters}
        predicted_increment = self._make_prediction_numpy(self.unfold_params(sample), roi, weights)

        weight_inverse = self.weight_inverse[roi["xmin"]: roi["xmax"], roi["ymin"]: roi["ymax"]]
        cost_kwarg = {
            "weights": weight_inverse
        }

        cost = self._calculate_cost(measured_increment, predicted_increment, **cost_kwarg)
        logger.info(f"{trial.number = } / {cost = }")
        return cost

    def _sampling(self, trial: optuna.Trial, key: str):
        return trial.suggest_float(
            key,
            self._opt_config["parameters"][key]["min"],
            self._opt_config["parameters"][key]["max"],
        )

    def _run_scipy(self, measured_increment: np.ndarray, roi: dict,
                   weights: np.ndarray = None):
        """Runs optimization of (v_x, v_y) for one patch

        Args:
            measured_increment: cropped and normed patch of the event histogram

        Returns:
            (v_x, v_y)
        """
        callback = lambda pms: self._scipy_optimization_callback(pms,
                                                                 measured_increment,
                                                                 roi,
                                                                 weights)
        x0 = torch.from_numpy(self._initialize_velocity()).double().to(self._device)
        measured_increment = torch.from_numpy(measured_increment).double().to(self._device)
        weights = torch.from_numpy(weights).double().to(self._device) if weights is not None else weights
        result = scipy_autograd.minimize(
            lambda x: self._objective_scipy(x, measured_increment, roi, weights),
            x0=x0,
            method=self._opt_method,
            options={"gtol": 1e-8, "disp": True},
            callback=callback,
        )
        if not result.success:
            logger.warning("Unsuccessful optimization step!")
        return result

    def _run_torch(self, measured_increment: np.ndarray, roi: dict,
                   weights: np.ndarray = None):
        """Runs optimization of (v_x, v_y) for one patch

        Args:
            measured_increment: cropped and normed patch of the event histogram

        Returns:
            (v_x, v_y)
        """
        x0 = torch.from_numpy(self._initialize_velocity()).double().to(self._device).requires_grad_()
        measured_increment = torch.from_numpy(measured_increment).double().to(self._device).requires_grad_()
        weights = torch.from_numpy(weights).double().to(
            self._device).requires_grad_() if weights is not None else weights

        lr_step = iters = self._opt_config["n_iter"]
        # lr, lr_decay = 0.05, 0.1
        lr, lr_decay = 0.01, 0.1
        optimizer = torch.optim.__dict__[self._opt_method]([x0], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)
        best_x, best_it, min_loss = x0, 0, math.inf
        for it in range(iters):
            optimizer.zero_grad()
            loss = self._objective_scipy(x0, measured_increment, roi, weights)
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
        return {"param": best_x.detach().cpu().numpy(), "loss": min_loss, "best_iter": best_it}

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
        roi_dense_flow = self._extrapolate_dense_flow_from_estimates(parameters, roi)[:, roi["xmin"]: roi["xmax"],
                         roi["ymin"]: roi["ymax"]]
        weight_inverse = self.weight_inverse[roi["xmin"]: roi["xmax"], roi["ymin"]: roi["ymax"]]
        cost_kwarg = {"flow": roi_dense_flow,
                      "weights": torch.from_numpy(weight_inverse).double().to(self._device)}
        if self.do_weight_inverse:
            self._video_maker.visualize_image((weight_inverse*255).astype(np.uint8), "opt_mask")
        if self._gml_config["optimize_warp"]:
            translation = self._extrapolate_dense_translation_from_estimates(parameters)
            roi_translation = translation[:, roi["xmin"]: roi["xmax"], roi["ymin"]: roi["ymax"]]
            cost_kwarg.update({"pxy": roi_translation})
        if self.is_poisson_model:
            intensity = self._get_patch_poisson(parameters)
            cost_kwarg.update({"intensity": intensity})

        cost = self._calculate_cost(measured_increment, predicted_increment, **cost_kwarg)
        logger.debug(f"loss: {cost:.6f}")
        return cost

    def _scipy_optimization_callback(self, parameters, measured_increment, roi, weights):
        # For video maker - Only run for debug mode.
        if not logger.isEnabledFor(logging.DEBUG):
            return

        measured_increment = measured_increment.clone().detach().cpu().numpy()
        if weights is not None:
            weights = weights.clone().detach().cpu().numpy()

        prediction = self._make_prediction_numpy(self.unfold_scipy_params(parameters),
                                                 roi, weights)
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

    def _extrapolate_dense_flow_from_estimates(self, parameters: torch.Tensor, roi: dict):
        """

        Args:
            parameters:
            roi:

        Returns:

        """
        if self.is_angle_model:
            assert len(parameters) in [1, 3]
            v_x, v_y = torch.sin(parameters[0]), torch.cos(parameters[0])
        else:
            assert len(parameters) in [2, 4]
            v_x, v_y = parameters[0], parameters[1]

        dense_flow = torch.zeros((2,) + self.orig_image_shape).double().to(self._device)
        dense_flow[0] += v_x
        dense_flow[1] += v_y
        return dense_flow

    def _initialize_velocity(self):
        # TODO: other initialization options
        if self.is_angle_model:
            angle = np.pi
            radial = 1.0
            p_x, p_y = 0., 0.
            if self._gml_config["optimize_warp"]:
                logger.debug(f"EKLT optimizer initialization with hardcoded values: {angle, p_x, p_y}")
                return np.array([angle, p_x, p_y], dtype=np.float64)
            else:
                return np.array([angle], dtype=np.float64)
        elif self.is_poisson_model:
            base = np.random.random() * 2. - 1
            if self._gml_config["optimize_warp"]:
                p_x, p_y = 0., 0.
                return np.array([base, p_x, p_y], dtype=np.float64)
            else:
                return np.array([base], dtype=np.float64)
        else:
            v_x, v_y = 0., 0.
            p_x, p_y = 0., 0.
            logger.debug(f"EKLT optimizer initialization with hardcoded values: {v_x, v_y, p_x, p_y}")
            if self._gml_config["optimize_warp"]:
                return np.array([v_x, v_y, p_x, p_y], dtype=np.float64)
            else:
                return np.array([v_x, v_y], dtype=np.float64)

    def _calculate_cost(self, predicted_increment, measured_increment, *args, **kwargs):
        cost_arg = {"prediction": predicted_increment,
                    "measurement": measured_increment,
                    "omit_boundary": True}
        cost_arg.update(kwargs)
        return self.cost_func.calculate(cost_arg)

    def _make_prediction_torch(self, parameters: torch.Tensor, roi: dict, weights: torch.Tensor):
        assert self.is_angle_model
        v_x, v_y = torch.sin(parameters[0]), torch.cos(parameters[0])
        x_min, x_max = roi["xmin"], roi["xmax"]
        y_min, y_max = roi["ymin"], roi["ymax"]

        if self._gml_config["optimize_warp"]:
            p_x, p_y = parameters[1], parameters[2]
            translation = torch.Tensor([p_x, p_y])
            orig_size_gradient_x = utils.frame_utils.warp_image_torch(self._gradient_x_torch.clone(),
                                                                      translation)
            orig_size_gradient_y = utils.frame_utils.warp_image_torch(self._gradient_y_torch.clone(),
                                                                      translation)
        else:
            orig_size_gradient_x = self._gradient_x_torch.clone()
            orig_size_gradient_y = self._gradient_y_torch.clone()

        gradient_x = orig_size_gradient_x[x_min: x_max, y_min: y_max]
        gradient_y = orig_size_gradient_y[x_min: x_max, y_min: y_max]

        predicted_increment = v_x * gradient_x + v_y * gradient_y

        if self._gml_config["no_polarity"]:
            predicted_increment = np.abs(predicted_increment)

        if weights is not None:
            predicted_increment *= weights
        predicted_increment /= torch.linalg.norm(predicted_increment.clone()) + 0.0001
        return predicted_increment

    def _make_prediction_numpy(self, parameters: dict, roi: dict, weights: np.ndarray):
        """

        Args:
            parameters:
            weights:

        Returns:

        """
        v_x, v_y = parameters["v_x"], parameters["v_y"]
        orig_h, orig_w = self.orig_image_shape
        x_min, x_max = roi["xmin"], roi["xmax"]
        y_min, y_max = roi["ymin"], roi["ymax"]

        if self._gml_config["optimize_warp"]:
            p_x, p_y = parameters["p_x"], parameters["p_y"]
            homography = np.array([
                [1, 0, p_y],
                [0, 1, p_x],
                [0, 0, 1]], dtype=np.float64)
            orig_size_gradient_x = cv2.warpPerspective(self._gradient_x, homography,
                                                       (orig_w, orig_h))
            orig_size_gradient_y = cv2.warpPerspective(self._gradient_y, homography,
                                                       (orig_w, orig_h))
        else:
            orig_size_gradient_x = self._gradient_x
            orig_size_gradient_y = self._gradient_y

        gradient_x = orig_size_gradient_x[x_min: x_max, y_min: y_max]
        gradient_y = orig_size_gradient_y[x_min: x_max, y_min: y_max]

        predicted_increment = v_x * gradient_x + v_y * gradient_y

        if self._gml_config["no_polarity"]:
            predicted_increment = np.abs(predicted_increment)

        if weights is not None:
            predicted_increment *= weights
        predicted_increment /= np.linalg.norm(predicted_increment) + 0.0001
        return predicted_increment

    def _make_measured_increment(self, events: np.ndarray, roi: dict) -> np.ndarray:
        """Determines the brightness increment from the observed events including
        cropping and normalization. Additionally, returns the weight mask

        Args:
            events:

        Returns:

        """
        x_min, x_max = roi["xmin"], roi["xmax"]
        y_min, y_max = roi["ymin"], roi["ymax"]

        # TODO: atm, calculates histogram for the whole image for every time this function is called
        pol_image = self.orig_imager.create_iwe(events,
                                                method="polarity",
                                                sigma=0)

        if self._gml_config["no_polarity"]:
            histogram = pol_image[0] + pol_image[1]
        else:
            histogram = pol_image[0] - pol_image[1]  # positive - negative

        if self._gml_config["weight_loss_by_event_hist"]:
            weights = cv2.GaussianBlur(np.abs(histogram), ksize=None,
                                       sigmaX=self._gml_config["weight_sigma"])
            weights = weights[x_min: x_max, y_min: y_max]
            if logger.isEnabledFor(logging.DEBUG):
                self.visualizer.visualize_image(range_norm(weights, dtype=np.uint8),
                                                file_prefix="weights")
        else:
            weights = None

        if self._gml_config["iwe_sigma"]:
            histogram = cv2.GaussianBlur(histogram, ksize=None,
                                         sigmaX=self._gml_config["iwe_sigma"])

        measured_increment = histogram[x_min: x_max, y_min: y_max]
        if weights is not None:
            measured_increment = weights * measured_increment
        measured_increment /= np.linalg.norm(measured_increment)
        if logger.isEnabledFor(logging.DEBUG):
            self.visualizer.visualize_image(range_norm(measured_increment, dtype=np.uint8),
                                            file_prefix="hist")
        return measured_increment, weights

    def make_diff_plot(self, measured: np.ndarray, params: dict, roi: dict, weights: np.ndarray, name: str):
        # params = self.unfold_params({
        #     "angle": angle,
        #     "p_x": p_x,
        #     "p_y": p_y
        # })

        predicted = self._make_prediction_numpy(self.unfold_params(params), roi, weights)
        diff = predicted - measured

        d_min, d_max = np.min(diff), np.max(diff)
        lower, upper = self._gml_config["viz_diff_scale"]
        if d_min < lower:
            logger.warning(f"The lowest value in diff is {d_min} but lower scale is {lower}")
        if d_max > upper:
            logger.warning(f"The lowest value in diff is {d_max} but lower scale is {upper}")

        diff = range_norm(diff, lower=lower, upper=upper, dtype=np.uint8)
        predicted = range_norm(predicted, dtype=np.uint8)
        measured = range_norm(measured, dtype=np.uint8)

        image = np.concatenate((diff, predicted, measured), axis=1)
        self.visualizer.visualize_image(image, name)

    def _estimate_patch(self, events: np.ndarray, roi: dict) -> np.ndarray:
        """Estimate EKLT parameters for one patch

        Args:
            events:
            roi: dict containing the keys xmin, xmax, ymin, ymax

        Returns:
            (result, data_artifacts), the result is a dict containing the best
            parameters and the according best value. data_artifacts contains
            intermediate data representations for visualization and debugging
        """
        if events.shape[0] == 0:
            logger.warning("Calling eklt patch estimation with zero events!")
            result = {
                "p_x": 0, "p_y": 0, "angle": 0, "best_value": 1e10
            }
            return result, None

        measured_increment, weights = self._make_measured_increment(events, roi)

        if self._opt_method in SCIPY_OPTIMIZERS:
            scipy_result = self._run_scipy(measured_increment, roi, weights)
            if self.is_angle_model:
                result = {"angle": scipy_result.x[0],
                          "p_x": scipy_result.x[1], "p_y": scipy_result.x[2],
                          "best_value": scipy_result.fun}
            else:
                result = {"v_x": scipy_result.x[0], "v_y": scipy_result.x[1],
                          "p_x": scipy_result.x[2], "p_y": scipy_result.x[3],
                          "best_value": scipy_result.fun}
            if logger.isEnabledFor(logging.DEBUG):
                self.visualizer.visualize_scipy_history(self.cost_func.get_history())
        elif self._opt_method in TORCH_OPTIMIZERS:
            if self.is_angle_model:
                opt_result = self._run_torch(measured_increment, roi, weights)
                result = {"angle": opt_result["param"][0], "p_x": opt_result["param"][1], "p_y": opt_result["param"][2],
                          "best_value": opt_result["loss"]}
            else:
                opt_result = self._run_torch(measured_increment, roi, weights)
                result = {"v_x": opt_result["param"][0], "v_y": opt_result["param"][1],
                          "p_x": opt_result["param"][2], "p_y": opt_result["param"][3],
                          "best_value": opt_result["loss"]}
            if logger.isEnabledFor(logging.DEBUG):
                self.visualizer.visualize_scipy_history(self.cost_func.get_history())
        elif self._opt_method == "optuna":
            study = self._run_optuna(measured_increment, roi, weights)
            if logger.isEnabledFor(logging.DEBUG):
                self.visualizer.visualize_optuna_history(study)
                self.visualizer.visualize_optuna_study(study, params=self._opt_parameters,
                                                       file_prefix="slices")
            result = study.best_params
            result["best_value"] = copy.deepcopy(study.best_value)
            if logger.isEnabledFor(logging.DEBUG):
                self.make_diff_plot(measured_increment, result, roi, weights, "best_params")
        else:
            logger.error(f"The optimization method {self._opt_method} is not"
                         f"implemented for  the solver {self.slv_config['method']}")
            raise NotImplementedError

        logger.info(
            f"End optimization.\n Result: {result}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            self.save_flow_error_as_text(self.iter_cnt, result,
                                         f"best_values")
            self.iter_cnt += 1

        data_artifacts = {
            "measured_increment": measured_increment,
            "gradient_x": self._gradient_x,
            "gradient_y": self._gradient_y,
            "weights": weights,
        }

        self._video_maker.make_video()
        self.cost_func.clear_history()

        return result, data_artifacts

    def estimate(self, events: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Estimate flow between given frame and events.

                Args:
                    events:
                    *args:
                    **kwargs: key "background" must contain background frame

                Returns:

        """
        self._video_maker = LossVideosMaker(self.orig_image_shape,
                                            os.path.join(self.visualizer.save_dir, str(self.iter_cnt)),
                                            "optimization")
        if self._gml_config["model_image"] == "current":
            self._set_frame(kwargs["frame"])
        elif self._frame is None and self._gml_config["model_image"] == "background":
            self._set_frame(kwargs["background"])

        roi = {key: self.slv_config["filter"]["parameters"][key]
               for key in ["xmin", "xmax", "ymin", "ymax"]}
        patch_result, _ = self._estimate_patch(events, roi)
        # TODO: for now returning flow from angle
        flow = np.empty((2,) + self.orig_image_shape, dtype=np.float64)
        if self.is_angle_model:
            flow[0] = np.sin(patch_result["angle"])
            flow[1] = np.cos(patch_result["angle"])
        else:
            flow[0] = patch_result["v_x"]
            flow[1] = patch_result["v_y"]
        return flow
