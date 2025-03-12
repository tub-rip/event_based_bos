# Event-based Background-Oriented Schlieren

This is the official repository for [**Event-based Background-Oriented Schlieren**](https://doi.org/10.1109/TPAMI.2023.3328188), **IEEE T-PAMI 2023** by  
[Shintaro Shiba](http://shibashintaro.com/), [Friedhelm Hamann](https://friedhelmhamann.github.io/), [Yoshimitsu Aoki](https://aoki-medialab.jp/aokiyoshimitsu-en/) and [Guillermo Callego](http://www.guillermogallego.es).

<h2 align="left">
  
[Paper](https://arxiv.org/pdf/2311.00434) | [Video](https://youtu.be/Ev52n8KgxIU) | [Dataset](https://doi.org/10.14279/depositonce-19492) | [News](https://www.tu.berlin/en/about/profile/press-releases-news/schlierenfotografie)
</h2>

[![Event-based Background-Oriented Schlieren](docs/img/event_based_bos_pami23.jpg)](https://youtu.be/Ev52n8KgxIU)


If you use this work in your research, please cite it (see also [here](#citation)):

```bibtex
@Article{Shiba23pami,
  author        = {Shintaro Shiba and Friedhelm Hamann and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Event-based Background-Oriented Schlieren},
  journal       = {IEEE Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  year          = {2024},
  volume        = {46},
  number        = {4},
  pages         = {2011-2026},
  doi           = {10.1109/TPAMI.2023.3328188}
}
```

-------
# Setup

## Requirements

- python: 3.8.x, 3.9.x, 3.10.x

- [ffmpeg](https://ffmpeg.org/) ... This is necessary for `ffmpeg-python`.

### Tested environments

- Mac OS Monterey (both M1 and non-M1)
- Ubuntu (CUDA 11.1, 11.3, 11.8)
- PyTorch 1.9-1.12.1, or PyTorch 2.0.

## Installation

### Quick setup

Minimal quick setup is to use [venv](https://docs.python.org/3/library/venv.html).

```shell
python3 -m venv ~/work/venv
source ~/work/venv/bin/activate  # activate env
python3 -V  # make sure the version, 3.8 or higher
pip3 install scipy==1.9.1 optuna==2.10.1 opencv-python matplotlib plotly \
ffmpeg-python h5py hdf5plugin PyYAML Pillow sklearn scikit-image argparse openpiv \
torch torchvision black isort kaleido
```

### Full setup

You can use [poetry](https://python-poetry.org/docs/).

```shell
poetry install
```

This install dev dependencies (format, test) too.

## Download dataset

Please download the dataset and put into your local folder. You can download the files from [the official source](https://doi.org/10.14279/depositonce-19492) or [Google Drive](https://drive.google.com/drive/folders/1kMnCKRI6QE8AmtxXe_jU9DTjlMbldpZM?usp=sharing)
The structure of the folder is as follows:

```shell
(root)/datasets/
    CCS/   # fixed keyword: dataset recorded with the co-capture system (CCS).
        (sequence_name)/
            basler_0/
                frames.mp4          # frame data
                config.yaml
            prophesee_0/
                cd_events.raw       # event data in .raw format (Prophesee)
                events.hdf5         # event data converted in HDF5 format
                trigger_events.txt
            homography.txt          # calibration file
```

## EBOS dataset recorded with Co-Capture System (CCS)

The Prophesee camera provides `.raw` format file, which requires to install `OpenEB`.
We converted this file into hdf5 file formats for better compatibility among languages and OS.
So, there is no need to install OpenEB.

# Execution

## Run

An example is:

```shell
python3 bos_event.py --config_file ./configs/hot_plate1.yaml --eval
```

## Config

The configuration for each experiment is provided to the script through yaml files.

Please see the [readme](./configs/README.md).

## Development

Note you use poetry for development.

## Format

You can quickly run format of the codes:

```shell
make fmt
```

<!--
# Citation

If you use this work in your research, please cite it as follows:

```bibtex
@Article{Shiba23pami,
  author        = {Shintaro Shiba and and Friedhelm Hamann and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Event-based Background-Oriented Schlieren},
  booktitle     = {IEEE T-PAMI},
  pages         = {},
  doi           = {10.1109/TPAMI.2023.3328188},
  year          = 2023
}
```
-->

# Code Authors

- [@shiba24](https://github.com/shiba24)
- [@FriedhelmHamann](https://github.com/FriedhelmHamann)

## LICENSE

Please check [License](./LICENSE).

-------
# Additional Resources

* [CoCapture: GUI for recording with multi camera systems.](https://github.com/tub-rip/CoCapture)
* [Secrets of Event-Based Optical Flow (TPAMI 2024)](https://github.com/tub-rip/event_based_optical_flow)
* [Motion-prior Contrast Maximization (ECCV 2024)](https://github.com/tub-rip/MotionPriorCMax)
* [Event Collapse in Contrast Maximization Frameworks](https://github.com/tub-rip/event_collapse)
* [EVILIP: Event-based Image Reconstruction as a Linear Inverse Problem (TPAMI 2022)](https://github.com/tub-rip/event_based_image_rec_inverse_problem)
* [EPBA: Event-based Photometric Bundle Adjustment](https://github.com/tub-rip/epba)
* [Research page (TU Berlin, RIP lab)](https://sites.google.com/view/guillermogallego/research/event-based-vision)
* [Research page (Keio University, Aoki Media Lab)](https://aoki-medialab.jp/home-en/)
* [Course at TU Berlin](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision)
* [Survey paper](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
* [List of Resources](https://github.com/uzh-rpg/event-based_vision_resources)
