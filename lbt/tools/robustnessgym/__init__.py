RGSUBPOPULATION_REGISTRY = {}

import importlib
import sys
import inspect

from .base_subpopulation import BaseSubpopulation
from .robustnessgym import RG
from robustnessgym.slicebuilders.subpopulation import Subpopulation

# from lbt.tools.robustnessgym imort RG


def register_lbtsubpop(name):
    def register_subpop_cls(cls):
        if not issubclass(cls, BaseSubpopulation):
            raise ValueError(
                "Metric ({}: {}) must extend lbt.tools.robustnessgym.base_subpopulation".format(
                    name, cls.__name__
                )
            )
        RGSUBPOPULATION_REGISTRY[name] = cls
        return cls

    return register_subpop_cls


LBT_SUBPOPULATIONS = {
    "lbt_subpops": "lbt.tools.robustnessgym.lbt_subpopulations",
}

RG_SUBPOPULATIONS = {
    "hans": "robustnessgym.slicebuilders.subpopulations.hans",
    "phrase": "robustnessgym.slicebuilders.subpopulations.phrase",
}

for name, module_name in LBT_SUBPOPULATIONS.items():
    if module_name not in sys.modules:
        importlib.import_module(module_name)

for name, module_name in RG_SUBPOPULATIONS.items():
    for name, obj in inspect.getmembers(sys.modules[module_name]):
        if inspect.isclass(obj):
            if issubclass(obj, Subpopulation):
                RGSUBPOPULATION_REGISTRY[name] = obj
