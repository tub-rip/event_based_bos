"""isort:skip_file
"""

from .base import SolverBase
from .generative_max_likelihood import GenerativeMaximumLikelihood
from .patch_eklt import PatchEklt
from .patch_eklt_dependent import PatchEkltDependent
from .patch_eklt_pyramid2 import PatchEkltPyramid2

# List of supported solver - non DNN
collections = {
    "generative_max_likelihood": GenerativeMaximumLikelihood,
    "patch_eklt": PatchEklt,
    "patch_eklt_dependent": PatchEkltDependent,
    "patch_eklt_pyramid2": PatchEkltPyramid2,
}
