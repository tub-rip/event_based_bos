# Configuration


The top level of the config yaml file has 
- `data`, see [the details](#data).
- `output_dir` ...  Single element.
- `evaluation`, see [the details](#evaluation).
- `common_params`, see [the details](#common_params)
- `solver`, see [the details](#solver)
- `method` ... Frame-based estimation method
- `params_opencv_flow` ... Optical flow GT estimation parameters.
- `params_openpiv` ... Not used in the paper.


## data

| Field    | Example Value                         | Required | Description                               |
|----------|---------------------------------------|----------|-------------------------------------------|
| root     |                                       | True     | Root data irectory. If empty, it will be `./datasets/`.                    |
| dataset  | CCS                                   | True     | CSS or DAVIS               |
| sequence | c_2022-10-20_18-27-27_41-roi-refr-100 | True     | The name of the sequence.                              |
| height   | 720                                   | True     | Height of event camera. |
| width    | 1280                                  | True     | Width of event camera.                              |
| warp     | True                                  | Optional | True to warp frames into events.          |

## evaluation

| Field     | Example Value | Required | Description                                          |
|-----------|---------------|----------|------------------------------------------------------|
| metrics   |               |  No      | (Not used yet)                          |
| time_list |               |  Yes     | The list (you can specify multiple) of [`start_time`, `end_time`]. |

## common_params

| Field    | Example Value | Required | Description                                          |
|----------|---------------|----------|------------------------------------------------------|
| n_frames | 1             |  Yes     | How many frames to be evaluated (or estimated), for 120 fps Basler camera. 1 means the evaluation is run based on each frame.            |
| xmin     | 210           |  Yes     | Height direction, top of the ROI. Events will be filtered accordingly.                                         |
| xmax     | 510           |  Yes     | Height direction, bottom of the ROI.            |
| ymin     | 300           |  Yes     | Width direction, left of the ROI. |
| ymax     | 500           |  Yes     | Width direction, right of the ROI.                                         |

## solver

Probably under `method` section is important, when you want to try new algorithms.


| Field                     | Example Value         | Required | Description                                                                                    |
|---------------------------|-----------------------|----------|------------------------------------------------------------------------------------------------|
| filter                    |                       |          | .                                                                             |
|   filters                 | [BAF                  |          | List of the filters you want. Available values are [`BAF`, `HOT`]. Crop is automatically added by default, using `common_params`. |
|   parameters              |                       |          | Parameters for the each filter.                                          |
|     BAF_dt                | 0.005                 |          | In second. Applicable when you use `BAF` in the `filters`.                                                     |
|     BAF_ksize             | 3                     |          | Applicable when you use `BAF` in the `filters`.                                                           |
|     BAF_num_support_event | 2                     |          | Applicable when you use `BAF` in the `filters`.                                                           |
|     BAF_continuous_update | True                  |          | Applicable when you use `BAF` in the `filters`.                                                           |
|     HOT_thresh            | 10                    |          | Applicable when you use `HOT` in the `filters`.                                                                                   |
|                           |                       |          |                                                                                                |
| method                    | contrast_maximization |  Yes        | The method of optimization. Supported values are [here](../src/solver/__init__.py)             |
| warp_direction            | first                 |          | Applicable when CMax-based method is used. Choose [`first`, `middle`, `last`].                    |
| outer_padding             | 0                     |          | Lorem ipsum dolor sit amet, consecteteur adipiscing.                                           |
|                           |                       |          |                                                                                                |
| motion_model              | 2d-translation        |          | Applicable when CMax-based method is used. Supported vallues are [here](../src/warp.py)                  |
| parameters                |                       |          |          |
|                           |                       |          |                                                                                                |
| cost                      | hybrid                |          | Basically hybrid is fine, but you can choose from [here](../src/costs/__init__.py)      |
| cost_with_weight          |                       |          | Applicable when `cost` is `hybrid`. Then, the following parameters will be effective.                    |
|   `cost_name`          | 1.0                   |          | Effective when `cost` is `hybrid`. The cost name as the key, the weight (for the weighed sum of the total cost) as the value. Can be multple.    |
| iwe                       |                       |          |                    |
|   method                  | bilinear_vote         |          | Method to create IWE, usually `bilinear_vote` is fine.                                      |
|   blur_sigma              | 3                     |          | Sigma for IWE. Usually 1 to 3. For BOS data I'm using 2 or 3.                   |
| optimizer                 |                       |          |    |
|   method                  | optuna                |          | Optimizing method for the solver. Choose from `optuna`, scipt-based ones, and torch-based ones. The scipt-based and the torch-based ones are in [here](../src/solver/contrast_maximization.py). They are gradient-based methods and `optuna` is sampling method.  |
|   sampler                 | uniform               |          | Applicable when `method` is `optuna`. `TPE` (intelligent), `uniform`, and `random`          |
|   n_iter                  | 800                   |          | Applicable when `method` is `optuna`. The number of sampling.          |
|   parameters              |                       |          | The key of this section should match with the contents of `parameters`      |
|     trans_x               |                       |          |   |
|       min                 | -30                   |          | Min of the search area. This is displacement. Let's say you set `n_frames` in the [common_params](#common_params). Then, the `-30` is the pixel within the 8.3 milliseconds (1/120). |
|       max                 | 30                    |          | Max of the search area.  |
|     trans_y               |                       |          |   |
|       min                 | -200                  |          |   |
|       max                 | 200                   |          |   |


## method


This is the method for frame-based optical flow.


| Field  | Example Value | Required | Description                               |
|--------|---------------|----------|-------------------------------------------|
| method | opencv_flow   |          | `all`, `opencv_flow`, `openpiv` |



## params_opencv_flow


| Field      | Example Value | Required | Description                                                                                   |
|------------|---------------|----------|-----------------------------------------------------------------------------------------------|
| pyr_scale  | 0.5           |          | |
| levels     | 4             |          | |
| winsize    | 10            |          | |
| iterations | 3             |          | |
| poly_n     | 5             |          | |
| poly_sigma | 1.2           |          | |
| flags      | 0             |          | |


## params_openpiv

| Field              | Example Value | Required | Description                                                                                 |
|--------------------|---------------|----------|---------------------------------------------------------------------------------------------|
| deformation_method | symmetric     |          |  |
|                    |               |          |  |
| windowsizes        |               |          |  |
|                    |               |          |  |
| overlap            |               |          |  |
|                    |               |          |  |
| MinMax_U_disp      |               |          |  |
|                    |               |          |  |
| MinMax_V_disp      |               |          |  |

