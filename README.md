# Event-based Background-Oriented Schlieren (T-PAMI 2023)

This is the official repository for **Event-based Background-Oriented Schlieren**, **IEEE T-PAMI 2023** by  
[Shintaro Shiba](http://shibashintaro.com/), [Friedhelm Hamann](https://friedhelmhamann.github.io/), [Yoshimitsu Aoki](https://aoki-medialab.jp/aokiyoshimitsu-en/) and [Guillermo Callego](https://sites.google.com/view/guillermogallego).


[[Video](https://youtu.be/v6ms6g2eOB8)] [[Dataset](https://drive.google.com/file/d/1LjvrYUWBBAyiVzi3GJhlDaVA5gokCGBd/view?usp=sharing)] [[PDF]()]
 [[arXiv]()]

[![Event-based Background-Oriented Schlieren](docs/img/event_based_bos_pami23.jpg)](https://youtu.be/v6ms6g2eOB8)


If you use this work in your research, please cite it (see also [here](#citation)):

```bibtex
@Article{Shiba23pami,
  author        = {Shintaro Shiba and and Friedhelm Hamann and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Event-based Background-Oriented Schlieren},
  booktitle     = {IEEE T-PAMI},
  pages         = {},
  doi           = {},
  year          = 2023
}
```

-------
# Setup

## Requirements

- python: 3.8.x, 3.9.x, 3.10.x

### Tested environments

- Mac OS Monterey (both M1 and non-M1)
- Ubuntu (CUDA 11.1, 11.3, 11.8)
- PyTorch 1.9-1.12.1, or PyTorch 2.0.

## Installation

I strongly recommend to use venv: `python3 -m venv <new_venv_path>`
Also, you can use [poetry]().

- Install pytorch **< 1.13** or **>= 2.0** and torchvision for your environment. Make sure you install the correct CUDA version if you want to use it.

- If you use poetry, `poetry install`. If you use only venv, check dependecy libraries and install it from [here](./pyproject.toml).

- If you are having trouble to install pytorch with cuda using poetry refer to this [link](https://github.com/python-poetry/poetry/issues/6409). 

## Download dataset

Download each dataset under `./datasets` directory.

# Execution

TBD.

# Citation

If you use this work in your research, please cite it as follows:

```bibtex
@Article{Shiba23pami,
  author        = {Shintaro Shiba and and Friedhelm Hamann and Yoshimitsu Aoki and Guillermo Gallego},
  title         = {Event-based Background-Oriented Schlieren},
  booktitle     = {IEEE T-PAMI},
  pages         = {},
  doi           = {},
  year          = 2023
}
```

# Code Authors

- [@shiba24](https://github.com/shiba24)
- [@FriedhelmHamann](https://github.com/FriedhelmHamann)

## LICENSE

Please check [License](./LICENSE).

## Acknowledgement

TBD

-------
# Additional Resources

* [Research page (TU Berlin, RIP lab)](https://sites.google.com/view/guillermogallego/research/event-based-vision)
* [Research page (Keio University, Aoki Media Lab)](https://aoki-medialab.jp/home-en/)
* [Course at TU Berlin](https://sites.google.com/view/guillermogallego/teaching/event-based-robot-vision)
* [Survey paper](http://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
* [List of Resources](https://github.com/uzh-rpg/event-based_vision_resources)
