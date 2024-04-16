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
from .third_party.flow_former.core.utils.utils import InputPadder

logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


class FrameFlowEstimator(object):
    def __init__(self, visualizer_module: visualizer.Visualizer) -> None:
        self.rife_model_loaded = False
        self.flowformer_model_loaded = False
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

    def rife_eccv22(self, frame0, frame1, params_rife, **kwargs) -> np.ndarray:
        """Estimate flow from RIFE (ECCV 2022) pretrained model.
        Please see:
            - https://github.com/megvii-research/ECCV2022-RIFE
            - https://github.com/megvii-research/ECCV2022-RIFE/issues/278
        TODO: need to check standardization.

        Args:
            frame0 (_type_): [H, W]
            frame1 (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        if not self.rife_model_loaded:
            self.load_rife_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        I0 = (
                torch.from_numpy(frame0[None]).to(device, non_blocking=True).unsqueeze(0).float()
                / 255.0
        )
        I1 = (
                torch.from_numpy(frame1[None]).to(device, non_blocking=True).unsqueeze(0).float()
                / 255.0
        )

        input_frame = torch.cat(
            [I0.clone(), I0.clone(), I0.clone(), I1.clone(), I1.clone(), I1.clone()], dim=1
        )
        # I0 = torch.tensor(frame0).to(device)
        # I1 = torch.tensor(frame1).to(device)
        with torch.no_grad():
            # flow = model.flownet(I0, I1, timestep=1.0, returnflow=True)[:, :2] # will get flow1->0
            # flow = model.flownet(I0, I1, timestep=0.0, returnflow=True)[:, 2:4] # will get flow0->1
            crop_flow = self.model_rife.flownet(input_frame, timestep=0.0, returnflow=True)[
                        :, 2:4
                        ]  # will get flow0->1

        crop_flow = crop_flow.detach().cpu().numpy().squeeze()  # 2 x H x W
        pad_flow = utils.pad_to_same_resolution(crop_flow, params_rife, 0)
        return pad_flow

    def flowformer_eccv22(self, frame0, frame1, params_flowformer, **kwargs) -> np.ndarray:
        """Estimate flow from FlowFormer (ECCV 2022) pretrained model.
        Please see: https://github.com/drinkingcoder/FlowFormer-Official
        TODO: need to check standardization.

        Args:
            frame0 (_type_): [H, W]
            frame1 (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        if not self.flowformer_model_loaded:
            self.load_flowformer_model()

        image1 = torch.from_numpy(np.stack([frame0, frame0, frame0], axis=0)).float() / 255.0
        image2 = torch.from_numpy(np.stack([frame1, frame1, frame1], axis=0)).float() / 255.0

        with torch.no_grad():
            crop_flow = compute_flow(self.model_flowformer, image1, image2)  # Flow is [H, W, 2]

        crop_flow = crop_flow.squeeze().transpose((2, 0, 1))  # 2 x H x W
        # crop_flow[0] *= crop_flow.shape[1]
        # crop_flow[1] *= crop_flow.shape[2]
        pad_flow = utils.pad_to_same_resolution(crop_flow, params_flowformer, 0)
        return pad_flow

    # DNN model loaders
    def load_rife_model(self):
        logger.info("Loading RIFE model.")
        from .third_party.rife.model.RIFE import Model

        self.model_rife = Model(arbitrary=True)

        pretrained_model_path = os.path.join(ARTIFACT_DIR, "RIFE_m_train_log")
        if not os.path.exists(pretrained_model_path):
            e = f"Please download pretraied model into artifacts/ directory.\n You can download the model from https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view"
            logger.error(e)
            raise FileNotFoundError(e)

        self.model_rife.load_model(pretrained_model_path)
        self.model_rife.eval()
        self.model_rife.device()
        self.rife_model_loaded = True

    def load_flowformer_model(self):
        logger.info("Loading FlowFormer model.")
        from .third_party.flow_former.configs.submission import get_cfg
        from .third_party.flow_former.core.FlowFormer import build_flowformer

        cfg = get_cfg()
        cfg.model = os.path.join(ARTIFACT_DIR, cfg.model)
        if not os.path.exists(cfg.model):
            e = f"Please download pretraied model into artifacts/ directory as /flowformer/checkpoints.\n You can download the model from https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view"
            logger.error(e)
            raise FileNotFoundError(e)

        self.model_flowformer = torch.nn.DataParallel(build_flowformer(cfg))
        self.model_flowformer.load_state_dict(torch.load(cfg.model))

        if torch.cuda.is_available():
            self.model_flowformer.cuda()
        self.model_flowformer.eval()
        self.flowformer_model_loaded = True


# Code for FlowFormer
TRAIN_SIZE = [432, 960]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size})."
        )
    if image_shape[0] == TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_flow(model, image1, image2, weights=None) -> np.ndarray:
    image_size = image1.shape[1:]

    image1, image2 = image1[None], image2[None]
    if torch.cuda.is_available():
        image1, image2 = image1.cuda(), image2.cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:  # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:  # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h: h + TRAIN_SIZE[0], w: w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h: h + TRAIN_SIZE[0], w: w + TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (
                w,
                image_size[1] - w - TRAIN_SIZE[1],
                h,
                image_size[0] - h - TRAIN_SIZE[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow
