import os
from typing import Dict

DATASET_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets"
)

from .base import DataLoaderBase
from .ccs import CcsDataLoader
from .e2vid import E2vidDataLoader
from .helium import HeliumDataLoader


# List of supported dataset
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


collections: Dict[str, DataLoaderBase] = {k.NAME: k for k in inheritors(DataLoaderBase)}
