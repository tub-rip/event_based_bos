"""isort:skip_file
"""
from .base import CostBase
from .diff_norm import DifferenceNorm
from .flow_norm import FlowNorm
from .flow_norm_pxy import FlowNormPxy
from .image_gradient import ImageGradient

def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


functions = {k.name: k for k in inheritors(CostBase)}

# For hybrid loss
from .hybrid import HybridCost
