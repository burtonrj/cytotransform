from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "cytotransform.logicle_ext",
        [
            "cytotransform/logicle_ext/FastLogicle.cpp",
            "cytotransform/logicle_ext/Logicle.cpp"
        ]
    )
]

setup(
    name="cytotransform",
    version="0.1.0",
    ext_modules=ext_modules
)
