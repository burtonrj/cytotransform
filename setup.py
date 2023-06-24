from pybind11.setup_helpers import Pybind11Extension, build_ext
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

setup(
    name="cytotransform",
    version="0.1.2",
    author="Ross Burton",
    author_email="burtonrossj@gmail.com",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.11",
)
