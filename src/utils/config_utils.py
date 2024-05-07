import argparse
import logging
import os
import shutil
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from openpiv import windef

from .misc import fetch_runtime_information

logger = logging.getLogger(__name__)


# Config IO functions
def parse_args(default_path="./configs/scripts/davis.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default=default_path,
        help="Config file yaml path",
        type=str,
    )
    parser.add_argument(
        "--log", help="Log level: [debug, info, warning, error, critical]", type=str, default="info"
    )
    parser.add_argument(
        "--eval",
        help="Enable for evaluation run",
        action="store_true",
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    propagate_config(config)
    return config, args


def propagate_config(config: dict):
    """In-place function.
    There are some config parameters that are common among sections.
    Propagate such config parameters to ensure the same parameters.

    Args:
        config (dict): to be overloaded.
    """
    for key in ["xmin", "xmax", "ymin", "ymax"]:
        # copy to data loader
        config["data"][key] = config["common_params"][key]
        if "solver" in config.keys():
            # copy to event-based BOS
            config["solver"]["filter"]["parameters"][key] = config["common_params"][key]

    # Crop information
    config["data"]["crop_height"] = config["data"]["xmax"] - config["data"]["xmin"]
    config["data"]["crop_width"] = config["data"]["ymax"] - config["data"]["ymin"]

    pad_x0 = config["common_params"]["xmin"]
    pad_x1 = config["data"]["height"] - config["common_params"]["xmax"]
    pad_y0 = config["common_params"]["ymin"]
    pad_y1 = config["data"]["width"] - config["common_params"]["ymax"]
    pad_config = {
        "pad_x0": pad_x0,
        "pad_x1": pad_x1,
        "pad_y0": pad_y0,
        "pad_y1": pad_y1,
    }

    if "solver" in config.keys():
        config["solver"]["params_opencv_flow"] = config["params_opencv_flow"]
        config["solver"]["params_openpiv"] = config["params_openpiv"]

        config["solver"].update(pad_config)
        config["solver"]["crop_height"] = config["data"]["crop_height"]
        config["solver"]["crop_width"] = config["data"]["crop_width"]

    # evaluation
    if "evaluation" in config.keys():
        config["evaluation"]["dt"] = config["common_params"]["n_frames"]

    for k in ["opencv_flow", "openpiv", "rife", "flowformer"]:
        if f"params_{k}" in config.keys():
            config[f"params_{k}"].update(pad_config)
        else:
            config[f"params_{k}"] = pad_config


def save_config(save_dir: str, file_name: str, log_level: str = "INFO"):
    """Save configuration"""
    # Copy config file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(file_name, save_dir)
    # Fetch experimental information
    # runtime_config = fetch_runtime_information()
    # stream = open(os.path.join(save_dir, "runtime.yaml"), "w")
    # yaml.dump(runtime_config, stream)
    # Setup log level and log file
    log_level = getattr(logging, log_level, None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"{save_dir}/main.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# Config for OpenPIV
def load_config_openpiv(
    params_openpiv: dict, common_config: dict, save_dir: str
) -> windef.Settings:
    """Load config from dict and propagate it to OpenPIV config object.

    Args:
        config (dict): _description_

    Returns:
        windef.Settings: _description_
    """
    settings = windef.Settings()
    # Data related settings'
    # settings.filepath_images = ''  # Folder with the images to process
    settings.save_path = save_dir
    # settings.frame_pattern_b = 'frame_00000000.png'

    if "ROI" in params_openpiv.keys() and params_openpiv["ROI"] == "full":
        settings.ROI = "full"
    else:
        settings.ROI = [
            common_config["xmin"],
            common_config["xmax"],
            common_config["ymin"],
            common_config["ymax"],
        ]

    settings.deformation_method = params_openpiv["deformation_method"]
    # settings.deformation_method = 'second image'

    settings.windowsizes = tuple(params_openpiv["windowsizes"])
    settings.overlap = tuple(params_openpiv["overlap"])

    settings.num_iterations = len(settings.windowsizes)

    # settings.windowsizes = (128, 64, 32, 16, 8) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    # settings.overlap = (64, 32, 16, 8, 4) # This is 50% overlap

    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = "gaussian"

    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)

    # 'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    # settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    settings.sig2noise_threshold = 1.0
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = "peak2peak"
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    settings.sig2noise_validate = False

    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN

    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated

    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = False
    settings.scale_plot = 20  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings

    # 'Processing Parameters'
    settings.correlation_method = "circular"  # 'circular' or 'linear'
    # settings.normalized_correlation = True

    # 'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True

    settings.replace_vectors = True

    settings.filter_method = "localmean"
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 2
    settings.filter_kernel_size = 1  # kernel size for the localmean method

    settings.MinMax_U_disp = params_openpiv["MinMax_U_disp"]
    settings.MinMax_V_disp = params_openpiv["MinMax_V_disp"]

    # The second filter is based on the global STD threshold
    settings.std_threshold = 5  # threshold of the std validation

    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 5  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size = 2  # defines the size of the local median, it'll be 3 x 3

    # New settings for version 0.23.2c
    settings.image_mask = False

    # Image mask properties
    settings.dynamic_masking_method = None
    # settings.dynamic_masking_method = 'intensity'
    settings.dynamic_masking_threshold = 0.1
    settings.dynamic_masking_filter_size = 21

    # Smoothing after the first pass
    settings.smoothn = True  # Enables smoothing of the displacemenet field
    settings.smoothn_p = 0.05  # This is a smoothing parameter

    settings.show_all_plots = False

    settings.invert = False

    # settings.remove_mean_shift = False
    return settings
