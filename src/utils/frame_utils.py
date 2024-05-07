import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# Code obtained from https://github.com/alexlib/openpiv_bos_velocimetry/blob/master/1_Example_BOS_PIV.ipynb
from matplotlib import pyplot as plt
from openpiv import filters, preprocess, pyprocess, scaling, smoothn, tools, validation, windef
from openpiv.tools import transform_coordinates
from openpiv.windef import first_pass, multipass_img_deform
#from pivpy import graphics, io, pivpy
from skimage.util import invert

from ..types import FLOAT_TORCH, NUMPY_TORCH, is_numpy, is_torch

logger = logging.getLogger(__name__)


def standardize_image_minmax(
    array: np.ndarray, new_min: float = 0.0, new_max: float = 255
) -> np.ndarray:
    """Standardize image, given min and max. Note it does not change dtype.

    Args:
        array (np.ndarray): Image array
        new_min (float, optional): Standardized so that the new min is this value. Defaults to 0.0.
        new_max (float, optional): Standardized so that the new max is this value. Defaults to 255.

    Returns:
        np.ndarray: Standardized image array.
    """
    st = (array - array.min()) / (array.max() - array.min())  # 0 to 1
    return st * (new_max - new_min) + new_min


def standardize_image_center(
    array: np.ndarray, old_center: float = 0, new_center: float = 128, new_max: float = 255
) -> np.ndarray:
    """Standardize image, given the center value. Note it does not change dtype.

    Args:
        array (np.ndarray): Image array
        new_min (float, optional): Standardized so that the new center is this value. Defaults to 0.0.
        new_max (float, optional): Standardized so that the new max is this value. Defaults to 255.

    Returns:
        np.ndarray: Standardized image array.
    """
    max_abs = np.abs(array).max()
    return (array - old_center) / max_abs * (new_max - new_center) + new_center


def warp_image_forward(im1: NUMPY_TORCH, forward_flow: NUMPY_TORCH) -> NUMPY_TORCH:
    """Warp image using forward flow.

    Args:
        im1 (np.ndarray): [H, W]
        forward_flow (np.ndarray): [2, H, W]

    Returns:
        np.ndarray: [H, W]
    """
    if is_numpy(im1):
        im1_tensor = torch.from_numpy(im1.astype(np.float64))[None, None]  # b=1, c-=1, h, w
        flow_tensor = torch.from_numpy(forward_flow.astype(np.float64))[None]  # b=1, c=2, h, w
        _return_numpy = True
    else:
        im1_tensor = im1[None, None]
        flow_tensor = forward_flow[None]
        _return_numpy = False

    h, w = im1.shape
    coord_x, coord_y = torch.meshgrid(torch.arange(h), torch.arange(w))
    coord_x = coord_x[None, None] / ((h - 1) / 2.0) - 1
    coord_y = coord_y[None, None] / ((w - 1) / 2.0) - 1
    warp_x = coord_x.to(flow_tensor.device) - flow_tensor[:, [0]] / ((h - 1) / 2.0)
    warp_y = coord_y.to(flow_tensor.device) - flow_tensor[:, [1]] / ((w - 1) / 2.0)

    grid = torch.cat([warp_y, warp_x], dim=1).permute((0, 2, 3, 1))

    warped_im1 = torch.nn.functional.grid_sample(
        im1_tensor, grid, mode="bilinear", align_corners=True
    )
    if _return_numpy:
        return warped_im1.detach().cpu().numpy().squeeze()
    return warped_im1.squeeze()


def warp_image_torch(im1: torch.Tensor, global_shift: torch.Tensor) -> np.ndarray:
    """Warp image using global shift (translation)

    Args:
        im1 (torch.Tensor): [H, W]
        global_shift (torch.Tensor): [2]

    Returns:
        torch.Tensor: [H, W]
    """
    im1_tensor = im1[None, None]  # b=1, c-=1, h, w

    h, w = im1.shape
    coord_x, coord_y = torch.meshgrid(torch.arange(h), torch.arange(w))
    coord_x = coord_x[None, None] / ((h - 1) / 2.0) - 1
    coord_y = coord_y[None, None] / ((w - 1) / 2.0) - 1
    warp_x = coord_x - global_shift[0] / ((h - 1) / 2.0)
    warp_y = coord_y - global_shift[1] / ((w - 1) / 2.0)

    grid = torch.cat([warp_y, warp_x], dim=1).double().permute((0, 2, 3, 1)).to(im1.device)
    warped_im1 = torch.nn.functional.grid_sample(
        im1_tensor, grid, mode="bilinear", align_corners=True
    )
    return warped_im1.squeeze()

def pad_to_same_resolution(array: NUMPY_TORCH, pad_config: dict, constant_value: float = 0.0):
    """Pad the array to the desired shape based on the config.
    Backcast the shape: padding happens at the last -2 and -1 axis.

    Args:
        array (NUMPY_TORCH): _description_
        pad_config (dict): Should have "pad_x0", "pad_x1", "pad_y0", "pad_y1".
        constant_value (float): _description_
    """
    current_shape = array.shape
    if is_torch(array):
        pad_shape = (
            pad_config["pad_y0"],
            pad_config["pad_y1"],
            pad_config["pad_x0"],
            pad_config["pad_x1"],
        )
        return torch.nn.functional.pad(array, pad_shape, mode="constant", value=constant_value)
    elif is_numpy(array):
        pad_shape = [(0, 0) for _ in range(len(current_shape))]
        pad_shape[-2] = (pad_config["pad_x0"], pad_config["pad_x1"])
        pad_shape[-1] = (pad_config["pad_y0"], pad_config["pad_y1"])
        return np.pad(array, tuple(pad_shape), constant_values=constant_value)


def pad_to_same_resolution_center(array, desired_shape, constant_value=0):
    """Pad the array in the center to the desired shape.
    Backcast the shape: if array.shape = (2, h, w) and the desired_shape = (H, W),
    the output will be (2, H, W).
    Args:
        array (_type_): _description_
        desired_shape (_type_): _description_
        constant_value (_type_): _description_
    """
    current_shape = array.shape
    if len(current_shape) != len(desired_shape):
        desired_shape = current_shape[: -len(desired_shape)] + desired_shape
    pad_shape = [
        ((j - i) // 2, (j - i) - (j - i) // 2) for i, j in zip(current_shape, desired_shape)
    ]
    return np.pad(array, tuple(pad_shape), constant_values=constant_value)


def bos_optical_flow(frame_a: np.ndarray, frame_b: np.ndarray, config: dict) -> np.ndarray:
    """Estimate pixel displacement between two given frames using OpenCV.

    Args:
        frame_a (np.ndarray): _description_
        frame_b (np.ndarray): _description_
        config (dict): Config parameter for cv2.calcOpticalFlowFarneback

    Returns:
        flow (np.ndarray): [H, W, 2] Optical flow (displacement).
    """
    flow = cv2.calcOpticalFlowFarneback(
        frame_a,
        frame_b,
        np.zeros(frame_a.shape + (2,)),
        config["pyr_scale"],
        config["levels"],
        config["winsize"],
        config["iterations"],
        config["poly_n"],
        config["poly_sigma"],
        config["flags"],
    )
    return flow


# Code obtained from https://github.com/alexlib/openpiv_bos_velocimetry/blob/master/1_Example_BOS_PIV.ipynb
def save_figure_to_numpy(fig):
    # save it to a numpy array.
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    # print('-=-=-=---=', data.shape, fig.canvas.get_width_height())
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data  # .transpose(2, 0, 1)


def piv_run(frame_a, frame_b, settings, counter=0):
    # "first pass"
    x, y, u, v, s2n = first_pass(frame_a, frame_b, settings)

    if settings.show_all_plots:
        plt.figure()
        plt.quiver(x, y, u, -v, color="b")
        plt.gca().invert_yaxis()
        plt.title("First path (displacement per window)")

    # " Image masking "
    if settings.image_mask:
        image_mask = np.logical_and(mask_a, mask_b)
        mask_coords = preprocess.mask_coordinates(image_mask)
        # mark those points on the grid of PIV inside the mask
        grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)

        # mask the velocity
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)
    else:
        mask_coords = []
        u = np.ma.masked_array(u, mask=np.ma.nomask)
        v = np.ma.masked_array(v, mask=np.ma.nomask)

    if settings.validation_first_pass:
        u, v, mask = validation.typical_validation(u, v, s2n, settings)

    if settings.show_all_plots:
        plt.figure()
        plt.quiver(x, y, u, -v, color="r")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1.0)
        plt.title("after first pass validation new, inverted")
        plt.show()

    # "filter to replace the values that where marked by the validation"
    if settings.num_iterations == 1 and settings.replace_vectors:
        # for multi-pass we cannot have holes in the data
        # after the first pass
        u, v = filters.replace_outliers(
            u,
            v,
            method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )
    elif settings.num_iterations > 1:  # don't even check if it's true or false
        u, v = filters.replace_outliers(
            u,
            v,
            method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )

        # "adding masks to add the effect of all the validations"
    if settings.smoothn:
        u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=settings.smoothn_p)
        v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=settings.smoothn_p)

    if settings.image_mask:
        grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)
    else:
        u = np.ma.masked_array(u, np.ma.nomask)
        v = np.ma.masked_array(v, np.ma.nomask)

    if settings.show_all_plots:
        plt.figure()
        plt.quiver(x, y, u, -v)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1.0)
        plt.title("before multi pass, inverted")
        plt.show()

    if not isinstance(u, np.ma.MaskedArray):
        raise ValueError("Expected masked array")

    """ Multi pass """

    for i in range(1, settings.num_iterations):

        if not isinstance(u, np.ma.MaskedArray):
            raise ValueError("Expected masked array")

        x, y, u, v, s2n, mask = multipass_img_deform(
            frame_a, frame_b, i, x, y, u, v, settings, mask_coords=mask_coords
        )

        # If the smoothing is active, we do it at each pass
        # but not the last one
        if settings.smoothn is True and i < settings.num_iterations - 1:
            u, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u, s=settings.smoothn_p)
            v, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v, s=settings.smoothn_p)
        if not isinstance(u, np.ma.MaskedArray):
            raise ValueError("not a masked array anymore")

        if hasattr(settings, "image_mask") and settings.image_mask:
            grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)
        else:
            u = np.ma.masked_array(u, np.ma.nomask)
            v = np.ma.masked_array(v, np.ma.nomask)

        if settings.show_all_plots:
            plt.figure()
            plt.quiver(x, y, u, -v, color="r")
            plt.gca().set_aspect(1.0)
            plt.gca().invert_yaxis()
            plt.title(f"end of the multipass (iteration {i}), invert")
            plt.show()

    if settings.show_all_plots and settings.num_iterations > 1:
        plt.figure()
        plt.quiver(x, y, u, -v)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1.0)
        plt.title("after multi pass, before saving, inverted")
        plt.show()

    # we now use only 0s instead of the image
    # masked regions.
    # we could do Nan, not sure what is best
    u = u.filled(0.0)
    v = v.filled(0.0)

    # "scales the results pixel-> meter"
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=settings.scaling_factor)

    if settings.image_mask:
        grid_mask = preprocess.prepare_mask_on_grid(x, y, mask_coords)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)
    else:
        u = np.ma.masked_array(u, np.ma.nomask)
        v = np.ma.masked_array(v, np.ma.nomask)

    # before saving we conver to the "physically relevant"
    # right-hand coordinate system with 0,0 at the bottom left
    # x to the right, y upwards
    # and so u,v

    x, y, u, v = transform_coordinates(x, y, u, v)
    # import pdb; pdb.set_trace()
    # "save to a file"
    tools.save(
        x,
        y,
        u,
        v,
        mask,
        os.path.join(settings.save_path, "field_A%03d.txt" % counter),
        delimiter="\t",
    )

    # "some other stuff that one might want to use"
    # if settings.show_plot or settings.save_plot:
    Name = os.path.join(settings.save_path, "Image_A%03d.png" % counter)
    fig, _ = display_vector_field(
        os.path.join(settings.save_path, "field_A%03d.txt" % counter),
        scale=settings.scale_plot,
    )
    if settings.save_plot is True:
        fig.savefig(Name)
    # if settings.show_plot is True:
    #     plt.show()
    return fig


# def func(args):
def piv(frame_a, frame_b, settings, counter=0):
    """A function to process each image pair."""
    # crop
    if settings.ROI == "full":
        frame_a = frame_a
        frame_b = frame_b
    else:
        frame_a = frame_a[settings.ROI[0] : settings.ROI[1], settings.ROI[2] : settings.ROI[3]]
        frame_b = frame_b[settings.ROI[0] : settings.ROI[1], settings.ROI[2] : settings.ROI[3]]

    if settings.invert is True:  # invert max / min of the image values
        frame_a = invert(frame_a)
        frame_b = invert(frame_b)

    if settings.show_all_plots:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(frame_a, cmap=plt.get_cmap("Reds"))
        ax.imshow(frame_b, cmap=plt.get_cmap("Blues"), alpha=0.5)
        plt.show()

    if settings.dynamic_masking_method in ("edge", "intensity"):
        frame_a, mask_a = preprocess.dynamic_masking(
            frame_a,
            method=settings.dynamic_masking_method,
            filter_size=settings.dynamic_masking_filter_size,
            threshold=settings.dynamic_masking_threshold,
        )
        frame_b, mask_b = preprocess.dynamic_masking(
            frame_b,
            method=settings.dynamic_masking_method,
            filter_size=settings.dynamic_masking_filter_size,
            threshold=settings.dynamic_masking_threshold,
        )

    fig = piv_run(frame_a, frame_b, settings, counter=counter)

    return fig


def display_vector_field(
    filename,
    on_img=False,
    image=None,
    window_size=32,
    scaling_factor=1,
    widim=False,
    ax=None,
    width=0.0025,
    **kw,
):
    """Displays quiver plot of the data stored in the file


    Parameters
    ----------
    filename :  string
        the absolute path of the text file

    on_img : Bool, optional
        if True, display the vector field on top of the image provided by
        image_name

    image : np.ndarray, optional
        image to plot the vector field onto when on_img is True

    window_size : int, optional
        when on_img is True, provide the interrogation window size to fit the
        background image to the vector field

    scaling_factor : float, optional
        when on_img is True, provide the scaling factor to scale the background
        image to the vector field

    widim : bool, optional, default is False
        when widim == True, the y values are flipped, i.e. y = y.max() - y

    Key arguments   : (additional parameters, optional)
        *scale*: [None | float]
        *width*: [None | float]


    See also:
    ---------
    matplotlib.pyplot.quiver


    Examples
    --------
    --- only vector field
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt',scale=100,
                                           width=0.0025)

    --- vector field on top of image
    >>> openpiv.tools.display_vector_field('./exp1_0000.txt', on_img=True,
                                          image_name='exp1_001_a.bmp',
                                          window_size=32, scaling_factor=70,
                                          scale=100, width=0.0025)

    """

    a = np.loadtxt(filename)
    # parse
    x, y, u, v, mask = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if on_img is True:  # plot a background image
        assert image is not None
        im = 255 - image  # plot negative of the image for more clarity
        xmax = np.amax(x) + window_size / (2 * scaling_factor)
        ymax = np.amax(y) + window_size / (2 * scaling_factor)
        ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    invalid = mask.astype("bool")
    valid = ~invalid

    # visual conversion for the data on image
    # to be consistent with the image coordinate system

    # if on_img:
    #     y = y.max() - y
    #     v *= -1

    if len(x[invalid]) > 0:
        ax.quiver(x[invalid], y[invalid], u[invalid], v[invalid], color="r", width=width, **kw)
    ax.quiver(x[valid], y[valid], u[valid], v[valid], color="b", width=width, **kw)

    # if on_img is False:
    #     ax.invert_yaxis()
    ax.set_aspect(1.0)
    # fig.canvas.set_window_title('Vector field, '+str(np.count_nonzero(invalid))+' wrong vectors')

    return fig, ax


def range_norm(matrix, new_max=255, lower=None, upper=None, dtype=None):
    if lower is None:
        lower = np.min(matrix)
    if upper is None:
        upper = np.max(matrix)

    matrix = np.clip(matrix, lower, upper)

    scaled = new_max * ((matrix - lower) / (upper - lower))
    if dtype is not None:
        scaled = scaled.astype(dtype)
    return scaled
