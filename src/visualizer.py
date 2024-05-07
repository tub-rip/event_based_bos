import glob
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import ffmpeg
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


from PIL import Image

from . import event_image_converter, types, warp
from .utils import poisson_reconstruct, standardize_image_center

TRANSPARENCY = 0.25  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)


class Visualizer:
    """Visualization class for multi utility. It includes visualization of
     - Events (polarity-based or event-based, 2D or 3D, etc...)
     - Images
     - Optical flow
     - Optimization history, loss function
     - Matplotlib figure
     etc.
    Also it works generic for if it saves and/or shows the figures.

    Args:
        image_shape (tuple) ... [H, W]. Image shape is necessary to visualize events.
        show (bool) ... It True, it shows the visualization results when any fuction is called.
        save (bool) ... It True, it saves the results under `save_dir` without any duplication.
        save_dir (str) ... Applicable when `save` is True. The root directory for the save.

    """

    def __init__(self, image_shape: tuple, show=False, save=False, save_dir=None) -> None:
        super().__init__()
        self.update_image_shape(image_shape)
        self._show = show
        self._save = save
        if save_dir is None:
            save_dir = "./"
        self.update_save_dir(save_dir)
        self.default_prefix = ""  # default file prefix
        self.default_save_count = 0  # default save count
        self.prefixed_save_count: Dict[str, int] = {}

    def update_image_shape(self, image_shape):
        self._image_size = image_shape  # H, W
        self._image_height = image_shape[0]
        self._image_width = image_shape[1]
        self.imager = event_image_converter.EventImageConverter(image_shape)

    def update_save_dir(self, new_dir: str) -> None:
        """Update save directiry. Creates it if not exist.

        Args:
            new_dir (str): New directory
        """
        self.save_dir = new_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_filename_from_prefix(
        self, prefix: Optional[str] = None, file_format: str = "png"
    ) -> str:
        """Helper function: returns expected filename from the prefix.
        It makes sure to save the output filename without any duplication.

        Args:
            prefix (Optional[str], optional): Prefix. Defaults to None.
            format (str) ... file format. Defaults to png.

        Returns:
            str: ${save_dir}/{prefix}{count}.png. Count automatically goes up.
        """
        if prefix is None or prefix == "":
            file_name = os.path.join(
                self.save_dir, f"{self.default_prefix}{self.default_save_count}.{file_format}"
            )
            self.default_save_count += 1
        else:
            try:
                self.prefixed_save_count[prefix] += 1
            except KeyError:
                self.prefixed_save_count[prefix] = 0
            file_name = os.path.join(
                self.save_dir, f"{prefix}{self.prefixed_save_count[prefix]}.{file_format}"
            )
        return file_name

    def rollback_save_count(self, prefix: Optional[str] = None):
        """Helper function:
        # hack - neeeds to be consistent number between .png and .npy

        Args:
            prefix (Optional[str], optional): Prefix. Defaults to None.
        """
        if prefix is None or prefix == "":
            self.default_save_count -= 1
        else:
            try:
                self.prefixed_save_count[prefix] -= 1
            except KeyError:
                raise ValueError("The visualization save count error")

    def reset_save_count(self, file_prefix: Optional[str] = None):
        if file_prefix is None or file_prefix == "":
            self.default_save_count = 0
        elif file_prefix == "all":
            self.default_save_count = 0
            self.prefixed_save_count = {}
        else:
            del self.prefixed_save_count[file_prefix]

    def _show_or_save_image(
        self, image: Any, file_prefix: Optional[str] = None, fixed_file_name: Optional[str] = None
    ):
        """Helper function - save and/or show the image.

        Args:
            image (Any): PIL.Image
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.
        """
        if self._show:
            if image.mode == "RGBA":
                image = image.convert("RGB")  # Back to RGB
            image.show()
        if self._save:
            if image.mode == "RGBA":
                image = image.convert("RGB")  # Back to RGB
            if fixed_file_name is not None:
                image.save(os.path.join(self.save_dir, f"{fixed_file_name}.png"))
            else:
                image.save(self.get_filename_from_prefix(file_prefix))

    def _show_or_save_plt(self, file_prefix: Optional[str] = None):
        """Helper function - save and/or show matplotlib figure.

        Args:
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.
        """
        if self._save:
            plt.savefig(self.get_filename_from_prefix(file_prefix))
        if self._show:
            plt.show(block=False)

    # Image related
    def load_image(self, image: Any) -> Image.Image:
        """A wrapper function to get image and returns PIL Image object.

        Args:
            image (str or np.ndarray): If it is str, open and load the image.
            If it is numpy array, it converts to PIL.Image.

        Returns:
            Image.Image: PIl Image object.
        """
        if type(image) == str:
            image = Image.open(image)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)
        return image

    def visualize_image(self, image: Any, file_prefix: Optional[str] = None) -> Image.Image:
        """Visualize image.

        Args:
            image (Any): str, np.ndarray, or PIL Image.
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.

        Returns:
            Image.Image: PIL Image object
        """
        image = self.load_image(image)
        self._show_or_save_image(image, file_prefix)
        return image

    def create_clipped_iwe_for_visualization(self, events, max_scale=50):
        """Utility function for clipped IWE. Same one in solver.

        Args:
            events (_type_): _description_
            max_scale (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        im = self.imager.create_image_from_events_numpy(events, method="bilinear_vote", sigma=0)
        clipped_iwe = 255 - np.clip(max_scale * im, 0, 255).astype(np.uint8)
        return clipped_iwe

    # Optical flow
    # Only flow
    def visualize_optical_flow(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
        visualize_color_wheel: bool = True,
        file_prefix: Optional[str] = None,
        save_flow: bool = False,
        ord: float = 0.5,
    ):
        """Visualize optical flow.
        Args:
            flow_x (numpy.ndarray) ... [H x W], height direction.
            flow_y (numpy.ndarray) ... [H x W], width direction.
            visualize_color_wheel (bool) ... If True, it also visualizes the color wheel (legend for OF).
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.

        Returns:
            image (PIL.Image) ... PIL image.
        """
        if save_flow:
            save_name = self.get_filename_from_prefix(file_prefix).replace("png", "npy")
            np.save(save_name, np.stack([flow_x, flow_y], axis=0))
            self.rollback_save_count(file_prefix)
        flow_rgb, color_wheel, _ = self.color_optical_flow(flow_x, flow_y, ord=ord)
        image = Image.fromarray(flow_rgb)
        self._show_or_save_image(image, file_prefix)

        if visualize_color_wheel:
            wheel = Image.fromarray(color_wheel)
            self._show_or_save_image(wheel, fixed_file_name="color_wheel")
        return image

    # Combined with events
    def visualize_overlay_optical_flow_on_event(
        self,
        flow: np.ndarray,
        events: np.ndarray,
        file_prefix: Optional[str] = None,
        ord: float = 0.5,
    ):
        """Visualize optical flow on event data.
        Args:
            flow (numpy.ndarray) ... [2 x H x W]
            events (np.ndarray) ... event_image (H x W) or raw events (n_events x 4).
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.

        Returns:
            image (PIL.Image) ... PIL image.
        """
        _show, _save = self._show, self._save
        self._show, self._save = False, False
        flow_image = self.visualize_optical_flow(flow[0], flow[1], ord=ord)
        flow_ratio = 0.8
        flow_image.putalpha(int(255 * flow_ratio))
        if events.shape[1] == 4:  # raw events
            event_image = self.visualize_event(events, grayscale=False).convert("RGB")
        else:
            event_image = self.visualize_image(events).convert("RGB")
        event_image.putalpha(255 - int(255 * flow_ratio))
        flow_image.paste(event_image, None, event_image)
        self._show, self._save = _show, _save
        self._show_or_save_image(flow_image, file_prefix)
        return flow_image

    def visualize_optical_flow_on_event_mask(
        self,
        flow: np.ndarray,
        events: np.ndarray,
        file_prefix: Optional[str] = None,
        ord: float = 0.5,
        max_color_on_mask: bool = True,
        mask_color: str = "white",
        mask_morph: bool = False,
    ):
        """Visualize optical flow only where event exists.
        Args:
            flow (numpy.ndarray) ... [2 x H x W]
            events (np.ndarray) ... [n_events x 4]
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.
            max_color_on_mask (bool) ... If True, the max magnitude is based on the masked flow. If False, it is based on the raw (dense) flow.
            mask_color (str) ... 'white', 'black'
            mask_morph (bool) ... If True, apply Close operation on the event mask.
                See https://docs.opencv.org/4.6.0/d3/dbe/tutorial_opening_closing_hats.html.

        Returns:
            image (PIL.Image) ... PIL image.
        """
        _show, _save = self._show, self._save
        self._show, self._save = False, False
        mask = self.imager.create_eventmask(events)
        if mask_morph:
            morph_size = 1
            # morph_elem = cv2.MORPH_RECT
            morph_elem = cv2.MORPH_CROSS
            # morph_size = 2
            # morph_elem = cv2.MORPH_ELLIPSE
            # morph_size = 3
            # morph_elem = cv2.MORPH_RECT

            element = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
            mask = cv2.morphologyEx(mask.astype(np.uint8)[0], cv2.MORPH_CLOSE, element).astype(bool)[None]        
        if max_color_on_mask:
            masked_flow = flow * mask
            image = self.visualize_optical_flow(
                masked_flow[0],
                masked_flow[1],
                visualize_color_wheel=False,
                file_prefix=file_prefix,
                ord=ord,
            )
        else:
            image = self.visualize_optical_flow(
                flow[0], flow[1], visualize_color_wheel=False, file_prefix=file_prefix, ord=ord
            )
        mask = Image.fromarray((~mask)[0]).convert("1")
        if mask_color == "white":
            white = Image.new("RGB", image.size, (255, 255, 255))
            masked_flow = Image.composite(white, image, mask)
        else:
            black = Image.new("RGB", image.size, (0, 0, 0))
            masked_flow = Image.composite(black, image, mask)
        self._show, self._save = _show, _save
        self._show_or_save_image(masked_flow, file_prefix)
        return masked_flow

    def visualize_optical_flow_pred_and_gt(
        self,
        flow_pred: np.ndarray,
        flow_gt: np.ndarray,
        visualize_color_wheel: bool = True,
        pred_file_prefix: Optional[str] = None,
        gt_file_prefix: Optional[str] = None,
        ord: float = 0.5,
    ):
        """Visualize optical flow both pred and GT.
        Args:
            flow_pred (numpy.ndarray) ... [2, H x W]
            flow_gt (numpy.ndarray) ... [2, H x W]
            visualize_color_wheel (bool) ... If True, it also visualizes the color wheel (legend for OF).
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.

        Returns:
            image (PIL.Image) ... PIL image.
        """
        # get largest magnitude in both pred and gt
        _, _, max_pred = self.color_optical_flow(flow_pred[0], flow_pred[1], ord=ord)
        _, _, max_gt = self.color_optical_flow(flow_gt[0], flow_gt[1], ord=ord)
        max_magnitude = np.max([max_pred, max_gt])
        color_pred, _, _ = self.color_optical_flow(
            flow_pred[0], flow_pred[1], max_magnitude, ord=ord
        )
        color_gt, color_wheel, _ = self.color_optical_flow(
            flow_gt[0], flow_gt[1], max_magnitude, ord=ord
        )

        image = Image.fromarray(color_pred)
        self._show_or_save_image(image, pred_file_prefix)
        image = Image.fromarray(color_gt)
        self._show_or_save_image(image, gt_file_prefix)
        if visualize_color_wheel:
            wheel = Image.fromarray(color_wheel)
            self._show_or_save_image(wheel, fixed_file_name="color_wheel")

    def color_optical_flow(
        self, flow_x: np.ndarray, flow_y: np.ndarray, max_magnitude=None, ord=1.0
    ):
        """Color optical flow.
        Args:
            flow_x (numpy.ndarray) ... [H x W], height direction.
            flow_y (numpy.ndarray) ... [H x W], width direction.
            max_magnitude (float, optional) ... Max magnitude used for the colorization. Defaults to None.
            ord (float) ... 1: our usual, 0.5: DSEC colorinzing.

        Returns:
            flow_rgb (np.ndarray) ... [W, H]
            color_wheel (np.ndarray) ... [H, H] color wheel
            max_magnitude (float) ... max magnitude of the flow.
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        flows[np.isinf(flows)] = 0
        flows[np.isnan(flows)] = 0
        mag = np.linalg.norm(flows, axis=2) ** ord
        ang = (np.arctan2(flow_y, flow_x) + np.pi) * 180.0 / np.pi / 2.0
        ang = ang.astype(np.uint8)
        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 255
        if max_magnitude is None:
            max_magnitude = mag.max()
        hsv[:, :, 2] = (255 * mag / max_magnitude).astype(np.uint8)
        # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Color wheel
        hsv = np.zeros([flow_x.shape[0], flow_x.shape[0], 3], dtype=np.uint8)
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, flow_x.shape[0]), np.linspace(-1, 1, flow_x.shape[0])
        )
        mag = np.linalg.norm(np.stack((xx, yy), axis=2), axis=2)
        # ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
        ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
        hsv[:, :, 0] = ang.astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = (255 * mag / mag.max()).astype(np.uint8)
        # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        color_wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return flow_rgb, color_wheel, max_magnitude

    # Poisson integration
    def visualize_poisson_integration(self, flow: np.ndarray, file_prefix: Optional[str] = None):
        """Visualize Poisson integration. The boundary is set to be zero by default.

        Args:
            flow (np.ndarray): [2, H, W]
            file_prefix (Optional[str], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        poisson = poisson_reconstruct(flow[1], flow[0], np.zeros_like(flow[0]))
        centered = standardize_image_center(poisson).astype(np.uint8)
        return self.visualize_image(centered, file_prefix=file_prefix)


    # Event related
    def visualize_event(
        self,
        events: Any,
        grayscale: bool = True,
        background_color: int = 127,
        ignore_polarity: bool = False,
        file_prefix: Optional[str] = None,
    ) -> Image.Image:
        """Visualize event as image.
        # TODO the function is messy - cleanup.

        Args:
            events (Any): [description]
            grayscale (bool, optional): [description]. Defaults to True.
            background_color: int = 127: Background color when events are none
            backround (int, optional): Only effective when grayscale is True. Defaults to 127. If non-grayscale, it is 255.
            ignore_polarity (bool, optional): If true, crerate polarity-ignore image. Defaults to False.

        Returns:
            Optional[Image.Image]: [description]
        """
        if grayscale:
            image = np.zeros((self._image_size[0], self._image_size[1]))
        else:
            background_color = 255
            image = (
                np.ones((self._image_size[0], self._image_size[1], 3), dtype=np.uint8)
                * background_color
            )  # RGBA channel

        x = np.clip(events[:, 0], 0, self._image_size[0] - 1).astype(np.int32)
        y = np.clip(events[:, 1], 0, self._image_size[1] - 1).astype(np.int32)
        if grayscale:
            indice = (x, y)
            if ignore_polarity:
                pol = np.ones_like(events[:, 3], dtype=np.int16)
            else:
                if np.min(events[:, 3]) == 0:
                    pol = events[:, 3] * 2 - 1
                else:
                    pol = events[:, 3]
            np.add.at(image, indice, pol)            
            image = np.clip((image * 20 + background_color), 0, 255).astype(np.uint8)
            logger.debug(f"Visualization, {image.min()}, {image.max()}")
        else:
            colors = np.array([(255, 0, 0) if e[3] == 1 else (0, 0, 255) for e in events])
            image[x, y, :] = colors

        image = Image.fromarray(image)
        self._show_or_save_image(image, file_prefix)
        return image

    def save_array(
        self,
        array: np.ndarray,
        file_prefix: Optional[str] = None,
        new_prefix: bool = False,
    ) -> None:
        """Helper function to save numpy array. It belongs to this visualizer class
        because it associates with the naming rule of visualized files.

        Args:
            array (np.ndarray): Numpy array to save.
            file_prefix (Optional[str]): Prefix of the file. Defaults to None.
            new_prefix (bool): If True, rollback_save_count is skipped. Set to True if
                there is no correspondng .png file with the prefix. bDefaults to False.

        Returns:
            Optional[Image.Image]: [description]
        """
        save_name = self.get_filename_from_prefix(file_prefix).replace("png", "npy")
        np.save(save_name, array)
        if not new_prefix:
            self.rollback_save_count(file_prefix)


    # ffmpeg interfaces
    def visualize_sequential_images_as_video(
        self, prefix: Optional[str] = None, numformat: Optional[str] = None
    ):
        """Given the prefix, collect all the images with the prefix and creates the sequential video of it.
        It requires ffmpeg.
        """
        if prefix is None or prefix == "":
            prefix = self.default_prefix
            total_count = self.default_save_count
            save_file = os.path.join(self.save_dir, f"out.mp4")
        else:
            total_count = self.prefixed_save_count[prefix]
            save_file = os.path.join(self.save_dir, f"{prefix}.mp4")
        if numformat is None:
            ffmpeg.input(os.path.join(self.save_dir, f"{prefix}%d.png"), r=20, f="image2",).output(
                save_file, vframes=total_count, vcodec="libx264", crf=10, pix_fmt="yuv420p"
            ).run(overwrite_output=True)
        else:
            ffmpeg.input(
                os.path.join(self.save_dir, f"{prefix}%{numformat}d.png"),
                r=20,
                f="image2",
            ).output(
                save_file,
                # vf="select=not(mod(n + 1\,4))",
                vframes=total_count,
                vcodec="libx264",
                crf=10,
                pix_fmt="yuv420p",
            ).run(
                overwrite_output=True
            )

    def concat_videos(self, video_file_list: list, save_name: str, stack: str = "h") -> str:
        """Concatenate multiple videos horizontally or vertically.

        Args:
            video_file_list (list): list of strings
            save_name (str): OUtput save name.
            stack (str) ... "h" or "v".
        Returns:
            save_file_name
        """
        if stack == "h":
            stack = "hstack"
        elif stack == "v":
            stack = "vstack"
        else:
            raise NotImplementedError("Stack direction must be h or v.")
        assert len(video_file_list) > 1

        save_file = os.path.join(self.save_dir, f"{save_name}.mp4")

        def _load_video_and_drawtext(filename: str):
            if not filename.endswith(".mp4"):
                filename += ".mp4"
            stream = ffmpeg.input(os.path.join(self.save_dir, filename))
            stream = ffmpeg.drawtext(stream, text=filename, x="(w-text_w)/10", y="0")
            return stream

        # concat
        stream = _load_video_and_drawtext(video_file_list[0])
        for next_file in video_file_list[1:]:
            next_stream = _load_video_and_drawtext(next_file)
            stream = ffmpeg.filter([stream, next_stream], stack)

        stream = ffmpeg.output(stream, save_file)
        ffmpeg.run(stream, overwrite_output=True)
        return save_file

    # matplotlib figure
    def visualize_plt_figure(self, figure, file_prefix: Optional[str] = None):
        """Visualize plt figure.

        Args:
            figure: plt figure object.
            file_prefix (Optional[str], optional): [description]. Defaults to None.
                If specified, the save location will be `save_dir/{prefix}_{unique}.png`.
        """
        if self._save:
            figure.savefig(self.get_filename_from_prefix(file_prefix))
        plt.close()

    # Scipy history visualizer
    def visualize_scipy_history(self, cost_history: dict, cost_weight: Optional[dict] = None):
        """Visualizing scipy optimization history.

        Args:
            cost_history (dict): [description]
        """
        plt.figure()
        for k in cost_history.keys():
            if k == "loss" or cost_weight is None:
                plt.plot(np.array(cost_history[k]), label=k)
            else:
                plt.plot(np.array(cost_history[k]) * cost_weight[k], label=k)
        plt.legend()
        if self._save:
            plt.savefig(self.get_filename_from_prefix("optimization_steps"))
        if self._show:
            plt.show(block=False)
        plt.close()
