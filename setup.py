from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cytotransform.logicle_ext",
        [
            "cytotransform/logicle_ext/FastLogicle.cpp",
            "cytotransform/logicle_ext/Logicle.cpp",
        ],
    )
]

setup(name="cytotransform", version="0.1.1", ext_modules=ext_modules)
