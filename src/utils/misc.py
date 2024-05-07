import ast
import cProfile
import logging
import os
import pstats
import random
import socket
import subprocess
import uuid
from functools import wraps
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def fix_random_seed(seed=46) -> None:
    """Fix random seed"""
    logger.info("Fix random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_file_utils(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]


def fetch_runtime_information() -> dict:
    """Fetch information of the experiment at runtime.

    Returns:
        dict: _description_
    """
    config = {}
    config["commit"] = fetch_commit_id()
    config["server"] = get_server_name()
    return config


def fetch_commit_id() -> str:
    """Get the latest commit ID of the repository.

    Returns:
        str: _description_
    """
    return "none"


def get_server_name() -> str:
    """Get server name based on MAC address. It is hard-coded.

    Returns:
        str: _description_
    """
    return "unknown"


def read_flow_error_text(filename: str, abs_val: bool = False) -> tuple:
    """Read per-frame error file and returns it with statistics.

    Args:
        filename (str): [description]
        abs_val (bool): If True, calculate statistics etc on abs value.

    Returns:
        error_per_frame (dict): {key: np.ndarray}
        stats_over_frame (dict): {key: {"mean", "std", "min", "max", "n_data"}}
    """
    file = open(filename, "r")
    cnt = 0
    while 1:
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            line = line.replace("nan", "0.0")
            data = ast.literal_eval(line[line.find("::") + 2 : line.rfind("\n")])
            if cnt == 0:
                error_metrics_list = data.keys()
                error_per_frame: Dict[str, list] = {k: [] for k in error_metrics_list}
            for k in error_metrics_list:
                error_per_frame[k].append(data[k])
            cnt += 1

    error_per_frame = {k: np.array(error_per_frame[k]) for k in error_metrics_list}
    if abs_val:
        error_per_frame = {k: np.abs(v) for (k, v) in error_per_frame.items()}
    # Convert FWL to inverse
    for k in error_metrics_list:
        if "FWL" in k:
            error_per_frame[k] = 1.0 / error_per_frame[k]
        if k in ["1PE", "2PE", "3PE", "5PE", "10PE", "20PE"]:
            error_per_frame[k] *= 100.0
    # error_per_frame = {k: v[v > 0] for (k, v) in error_per_frame.items()}
    # error_per_frame = {k: v[9910:10710] for (k, v) in error_per_frame.items()}   # only for outdoor_day1
    # error_per_frame = {k: v[50:] for (k, v) in error_per_frame.items()}   # slider_depth to see the difference

    stats_over_frame: Dict[str, dict] = {k: {} for k in error_metrics_list}
    for k in error_metrics_list:
        metric = np.copy(error_per_frame[k])
        if k == "AE":
            metric = metric[metric != 0]
        stats_over_frame[k]["mean"] = np.mean(metric)
        stats_over_frame[k]["rms"] = np.sqrt(np.mean(np.square(metric)))
        stats_over_frame[k]["std"] = np.std(metric)
        stats_over_frame[k]["min"] = np.min(metric)
        stats_over_frame[k]["max"] = np.max(metric)
        stats_over_frame[k]["n_data"] = len(metric)
    return error_per_frame, stats_over_frame


def profile(output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/

    Usage is,
    ```
    @profile(output_file= ...)
    def your_function():
        ...
    ```
    Then you will get the profile automatically after the function call is finished.

    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


import optuna


class SingleThreadInMemoryStorage(optuna.storages.InMemoryStorage):
    """This is faster version of in-memory storage only when the study n_jobs = 1 (single thread).

    Args:
        optuna ([type]): [description]
    """

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: optuna.distributions.BaseDistribution,
    ) -> None:
        with self._lock:
            trial = self._get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)

            study_id = self._trial_id_to_study_id_and_number[trial_id][0]
            # Check param distribution compatibility with previous trial(s).
            if param_name in self._studies[study_id].param_distribution:
                optuna.distributions.check_distribution_compatibility(
                    self._studies[study_id].param_distribution[param_name], distribution
                )
            # Set param distribution.
            self._studies[study_id].param_distribution[param_name] = distribution

            # Set param.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution
