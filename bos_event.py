# Event-based BOS estimates dense optical flow (displacement).
# This corresponds to the temporal derivative of the density gradient.

import json
import logging
import sys

import cv2
import numpy as np
from tqdm import tqdm
from skimage.util import invert

sys.path.append("./")
sys.path.append("../")

import logging

from src import data_loader, frame_flow_estimator, solver, utils, visualizer, event_image_converter

# the supported method for optical flow evaluation. currently only opencv_flow.
SUPPORTED_EVALUATION_METHOD = ["opencv_flow", "opencv_flow_two_steps", "openpiv", "openpiv_two_steps"]
SUPPORTED_ESTIMATION_METHOD = ["solver", "openpiv"]


def validate_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Validate and preprocess image (crop).

    Returns:
        images (np.ndarray)
        ts (float) timestamp
    """
    image = image[..., config["xmin"] : config["xmax"], config["ymin"] : config["ymax"]]
    assert (
        image.shape[0] % 2 == 0
    ), f"Cropped height should be even number: current {config['xmin']} to {config['xmax']}"
    assert (
        image.shape[1] % 2 == 0
    ), f"Cropped width should be even number: current {config['ymin']} to {config['ymax']}"
    return image


def evaluate_flow_on_event_grids(config, loader, viz):
    openpiv_config_events = config["params_openpiv_events"]
    integration_time = openpiv_config_events["integration_time"]
    frame_distance = openpiv_config_events["frame_distance"]
    do_inversion = openpiv_config_events["do_inversion"]
    eval_config = config["evaluation"]
    common_config = config["common_params"]
    eval_dt = eval_config["dt"]
    time_indices = eval_config["time_list"]  # list of [start, end] timestamps
    i_frame = 0
    cropped_image_shape = (config["data"]["crop_height"], config["data"]["crop_width"])
    orig_image_shape = (config["data"]["height"], config["data"]["width"])
    frame_estimator = frame_flow_estimator.FrameFlowEstimator(viz)
    imager = event_image_converter.EventImageConverter(orig_image_shape)
    for time_inds in time_indices:
        logger.info(f"Evaluation between {time_inds}")
        ind_start = loader.time_to_image_index(time_inds[0]) + 1
        ind_end = loader.time_to_image_index(time_inds[1]) - eval_dt
        logger.info(f"Corresponding indices: {ind_start}, {ind_end}")
        for i1 in tqdm(range(ind_start, ind_end, eval_dt)):  # this is no sliding window in time.
            # Process frames for GT
            i2 = i1 + eval_dt
            im1, t1 = loader.load_image(i1)
            im2, t2 = loader.load_image(i2)
            frame1 = validate_image(im1, common_config)
            frame2 = validate_image(im2, common_config)
            logger.info(f"Frame {i1} to {i2} (image index {i_frame})")
            if frame1.shape != cropped_image_shape or frame2.shape != cropped_image_shape:
                logger.warning(f"Warning! The frame might be collapsed -- {i1 = }, {i2 = }")
                continue
            gt_flow = frame_estimator.opencv_farneback(
                frame1, frame2, config["params_opencv_flow"], visualize_frame=False
            )

            # Process events
            ind1 = loader.time_to_index(t1)  # event indice
            ind2 = loader.time_to_index(t2)
            logger.info(f"Event {ind1} to {ind2}")
            batch_for_gt = loader.load_event(max(ind1, 0), min(ind2, len(loader)))

            # Process events
            ind_cur_2 = loader.time_to_index(t1)
            ind_cur_1 = loader.time_to_index(t1 - integration_time)
            ind_next_2 = loader.time_to_index(t1 + frame_distance)
            ind_next_1 = loader.time_to_index(t1 + frame_distance - integration_time)

            events1 = loader.load_event(ind_cur_1, ind_cur_2)
            events2 = loader.load_event(ind_next_1, ind_next_2)

            hist1 = imager.create_image_from_events_numpy(events1, method='bilinear_vote', sigma=0)
            hist1 *= 255.0/hist1.max()
            if do_inversion:
                hist1 = invert(hist1)

            hist2 = imager.create_image_from_events_numpy(events2, method='bilinear_vote', sigma=0)
            hist2 *= 255.0/hist2.max()
            if do_inversion:
                hist2 = invert(hist2)

            flow, fig = frame_estimator.consecutive_openpiv(hist1, hist2, config)

            viz.visualize_optical_flow(flow[0], flow[1], file_prefix="event_flow_openpiv")
            viz.visualize_plt_figure(fig, file_prefix="event_flow_vector")
            viz.visualize_image(hist1.astype(np.uint8), file_prefix="hist1")
            viz.visualize_image(hist2.astype(np.uint8), file_prefix="hist2")


def evaluate_per_frames(config, loader, solv, viz):
    """Evaluate event-based method based on frame-based method.

    Args:
        eval_config (_type_): _description_
        loader (_type_): _description_
        solv (_type_): _description_
    """
    eval_config = config["evaluation"]
    common_config = config["common_params"]
    cropped_image_shape = (config["data"]["crop_height"], config["data"]["crop_width"])
    eval_dt = eval_config["dt"]
    time_indices = eval_config["time_list"]  # list of [start, end] timestamps
    i_frame = 0
    n_events = (
        config["data"]["n_events_per_batch"]
        if "n_events_per_batch" in config["data"].keys()
        else None
    )
    max_event_dt = (
        config["data"]["max_time_per_event_batch"]
        if "max_time_per_event_batch" in config["data"].keys()
        else None
    )

    frame_estimator = frame_flow_estimator.FrameFlowEstimator(viz)
    im0, _ = loader.load_image(0)
    frame0 = validate_image(im0, common_config)    
    special_case_remove_nose = utils.check_key_and_bool(config["data"], "remove_nose")

    for time_inds in time_indices:
        logger.info(f"Evaluation between {time_inds}")
        ind_start = loader.time_to_image_index(time_inds[0]) + 1
        ind_end = loader.time_to_image_index(time_inds[1]) - eval_dt
        logger.info(f"Corresponding indices: {ind_start}, {ind_end}")
        for i1 in tqdm(range(ind_start, ind_end)):  # always 1 sliding window in time 
            # Process frames for GT
            i2 = i1 + eval_dt
            im1, t1 = loader.load_image(i1)
            im2, t2 = loader.load_image(i2)
            frame1 = validate_image(im1, common_config)
            frame2 = validate_image(im2, common_config)
            logger.info(f"Frame {i1} to {i2} (image index {i_frame})")
            if frame1.shape != cropped_image_shape or frame2.shape != cropped_image_shape:
                logger.warning(f"Warning! The frame might be collapsed -- {i1 = }, {i2 = }")
                continue
            gt_flow = frame_estimator.estimate(
                config["method"], frame0, frame1, frame2, config
            )

            # Process events
            ind1 = loader.time_to_index(t1)  # event indice
            ind2 = loader.time_to_index(t2)
            logger.info(f"Event {ind1} to {ind2}")
            batch_for_gt = loader.load_event(max(ind1, 0), min(ind2, len(loader)))

            if max_event_dt is not None and t2 - t1 > max_event_dt:
                t2 = t1 + max_event_dt
                ind1 = loader.time_to_index(t1)  # event indice
                ind2 = loader.time_to_index(t2)
            if n_events is not None:
                if ind2 - ind1 < n_events:
                    logger.info(
                        f"Less events in one GT flow sequence. Events: {ind2-ind1} / Expected: {n_events}"
                    )
                    insufficient = n_events - (ind2 - ind1)
                    ind1 -= insufficient // 2
                    ind2 += insufficient // 2
                elif ind2 - ind1 > n_events:
                    logger.info(
                        f"Too many events in one GT flow sequence. Events: {ind2-ind1} / Expected: {n_events}"
                    )
                    # This is for Secrets paper etc
                    ind1 = ind2 - n_events
            # In case you want to use different events from GT flow one:
            batch_for_estimation = loader.load_event(max(ind1, 0), min(ind2, len(loader)))
            if special_case_remove_nose:
                logger.info("Remove nose for visualization..")
                batch_for_gt = utils.remove_event(batch_for_gt, 0, 120, 990, 1050)
                batch_for_estimation = utils.remove_event(batch_for_estimation, 0, 120, 990, 1050)
            gt_time_scale = t2 - t1
            filtered_batch, batch_time_scale = solv.preprocess(batch_for_estimation)

            estimation = solv.estimate(
                filtered_batch, gt_flow, frame=im1, background=im0
            )  # gt flow is used only for EventFrameScaleEstimation,
            # frame and background in GenerativeMaximumLikelihood  
            logger.debug(f"""Max: {estimation.max()}, {gt_flow.max()}
            Min: {estimation.min()}, {gt_flow.min()}
            Norm(x): {np.abs(estimation[0]).mean()}, {np.abs(gt_flow[0]).mean()}
            Norm(y): {np.abs(estimation[1]).mean()}, {np.abs(gt_flow[1]).mean()}""")

            # Visualization
            solv.visualize_original_sequential(batch_for_gt, filtered_batch)
            solv.visualize_flows(estimation * gt_time_scale / batch_time_scale, gt_flow)
            solv.visualize_pred_sequential(
                filtered_batch, estimation * gt_time_scale / batch_time_scale
            )
            solv.visualize_gt_sequential(filtered_batch, gt_flow)

            # Error calculation.
            flow_error_without_mask = solv.calculate_flow_error(estimation[:, common_config["xmin"]:common_config["xmax"], common_config["ymin"]:common_config["ymax"]],
                                                                gt_flow[:, common_config["xmin"]:common_config["xmax"], common_config["ymin"]:common_config["ymax"]])  # type: ignore
            solv.save_flow_error_as_text(i_frame, flow_error_without_mask, "flow_error_per_frame_without_mask.txt")  # type: ignore

            flow_error_with_mask = solv.calculate_flow_error(estimation[:, common_config["xmin"]:common_config["xmax"], common_config["ymin"]:common_config["ymax"]],
                                                             gt_flow[:, common_config["xmin"]:common_config["xmax"], common_config["ymin"]:common_config["ymax"]],
                                                             events=filtered_batch,
                                                             roi=common_config)  # type: ignore
            solv.save_flow_error_as_text(i_frame, flow_error_with_mask, "flow_error_per_frame_with_mask.txt")  # type: ignore
            solv.save_flow_error_as_text(i_frame, {"t1": t1, "t2": t2}, "timestamps_per_frame.txt")
            i_frame += 1


def estimate_sequential(config, loader, solv):
    """Evaluate event-based method based on frame-based method.

    Args:
        eval_config (_type_): _description_
        loader (_type_): _description_
        solv (_type_): _description_
    """
    eval_config = config["evaluation"]
    eval_dt = eval_config["dt"]
    sliding_window = 0.01   # in sec
    i_frame = 0
    time_indices = eval_config["time_list"]  # list of [start, end] timestamps
    for time_inds in time_indices:
        logger.info(f"Sequential estimation purely events between {time_inds}")
        # ind_start = loader.time_to_image_index(time_inds[0]) + 1
        # ind_end = loader.time_to_image_index(time_inds[1]) - eval_dt
        # logger.info(f"Corresponding indices: {ind_start}, {ind_end}")
        steps = np.arange(time_inds[0], time_inds[1], sliding_window)

        for t1 in tqdm(steps):
            t2 = t1 + eval_dt * 0.008
            print(f"From {t1} to {t2}")
            # i1 = loader.time_to_index(t1)
            # i2 = loader.time_to_index(t1 + integration_time)   # integration time

            # i2 = i1 + eval_dt
            # t1 = loader.image_index_to_time(i1)
            # t2 = loader.image_index_to_time(i2)
            ind1 = loader.time_to_index(t1)  # event indice
            ind2 = loader.time_to_index(t2)
            logger.info(f"Event {ind1} to {ind2}")
            batch = loader.load_event(max(ind1, 0), min(ind2, len(loader)))
            filtered_batch, batch_time_scale = solv.preprocess(batch)
            # estimation = solv.estimate(filtered_batch)
            # solv.set_previous_frame_best_estimation(estimation)
            solv.save_flow_error_as_text(i_frame, {"t1": t1, "t2": t2}, "timestamps_per_frame.txt")
            i_frame += 1

            # Visualization
            solv.visualize_original_sequential(batch, filtered_batch)
            # solv.visualize_pred_sequential(filtered_batch, estimation)


def accumulate_sequential(config, loader, solv):
    """Accumulate events.

    Args:
        eval_config (_type_): _description_
        loader (_type_): _description_
        solv (_type_): _description_
    """
    eval_config = config["evaluation"]
    eval_dt = eval_config["dt"]
    sliding_window = 0.01   # in sec
    i_frame = 0
    time_indices = eval_config["time_list"]  # list of [start, end] timestamps
    for time_inds in time_indices:
        logger.info(f"Sequential estimation purely events between {time_inds}")
        # ind_start = loader.time_to_image_index(time_inds[0]) + 1
        # ind_end = loader.time_to_image_index(time_inds[1]) - eval_dt
        # logger.info(f"Corresponding indices: {ind_start}, {ind_end}")
        steps = np.arange(time_inds[0], time_inds[1], sliding_window)

        pos_neg = np.zeros((2, ) + solv.orig_image_shape)
        filtered_pos_neg = np.zeros((2, ) + solv.orig_image_shape)
        for t1 in tqdm(steps):
            t2 = t1 + eval_dt * 0.008
            print(f"From {t1} to {t2}")
            ind1 = loader.time_to_index(t1)  # event indice
            ind2 = loader.time_to_index(t2)
            logger.info(f"Event {ind1} to {ind2}")
            batch = loader.load_event(max(ind1, 0), min(ind2, len(loader)))
            filtered_batch, batch_time_scale = solv.preprocess(batch)
            
            pos_neg += solv.orig_imager.create_iwe(batch, method='polarity')
            filtered_pos_neg += solv.orig_imager.create_iwe(filtered_batch, method='polarity')
            orig_img = utils.standardize_image_center(pos_neg[0] - pos_neg[1])
            # print('-=-=-=-', orig_img.min())
            solv.visualizer.visualize_image(orig_img.astype(np.uint8), file_prefix='orig')

            filtered_img = utils.standardize_image_center(filtered_pos_neg[0] - filtered_pos_neg[1])
            solv.visualizer.visualize_image(filtered_img.astype(np.uint8), file_prefix='filter')

            solv.save_flow_error_as_text(i_frame, {"t1": t1, "t2": t2}, "timestamps_per_frame.txt")
            i_frame += 1
            # Visualization
            # solv.visualize_original_sequential(batch, filtered_batch)
            # solv.visualize_pred_sequential(filtered_batch, estimation)

if __name__ == "__main__":
    # Setup and validate parameters
    config, args = utils.parse_args(default_path="./configs/scripts/davis.yaml")
    data_config = config["data"]
    save_dir = config["output_dir"]
    utils.save_config(save_dir, args.config_file, args.log.upper())

    # Setup objects
    logger = logging.getLogger(__name__)
    loader = data_loader.collections[data_config["dataset"]](config=data_config)
    loader.set_sequence(data_config["sequence"])

    orig_image_shape = (data_config["height"], data_config["width"])
    crop_image_shape = (data_config["crop_height"], data_config["crop_width"])
    viz = visualizer.Visualizer(orig_image_shape, save=True, show=False, save_dir=save_dir)

    # Solver - event BOS
    method_name = config["solver"]["method"]
    solv: solver.SolverBase = solver.collections[method_name](
        orig_image_shape,
        crop_image_shape,
        calibration_parameter=loader.load_calib(),
        solver_config=config["solver"],
        visualize_module=viz,
    )

    # Start processing.
    logger.info("Start BOS estimation.")
    if args.eval:  # For evalluation, we need frame-based BOS settings.
        logger.info(f"Evaluation: {config['evaluation']}")
        # Parameter check
        assert config["method"] in SUPPORTED_EVALUATION_METHOD
        assert config["estimation_method"] in SUPPORTED_ESTIMATION_METHOD
        if config["estimation_method"] == "openpiv":
            evaluate_flow_on_event_grids(config, loader, viz)
        else:
            evaluate_per_frames(config, loader, solv, viz)
    else:  # No evaluation, just running sequential BOS estimation.
        estimate_sequential(config, loader, solv)
        # accumulate_sequential(config, loader, solv)

    # Make video
    for v in solv.sequential_video_list:
        logger.info(f"Make video {v}...")
        viz.visualize_sequential_images_as_video(v)
    # viz.concat_videos(solv.sequential_video_list, "result")

    # Additional videos
    try:
        additional_list = ["original", "pred_flow", "gt_flow"]
        viz.concat_videos(additional_list, "flow_comparison")
        additional_list = ["original", "pred_masked", "gt_masked"]
        viz.concat_videos(additional_list, "flow_comparison_masked")
    except:
        pass

    try:
        additional_list = ["original", "original_filter"]
        viz.concat_videos(additional_list, "video_filter_effect")
    except:
        pass

    if args.eval:
        for fname in solv.evaluation_text_list:
            data, stat = utils.read_flow_error_text(fname)
            logger.info(f"Evaluation {fname}:\n{stat}")
