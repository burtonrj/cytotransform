from typing import Any, Dict

from setuptools_cpp import ExtensionBuilder, Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "logicle_ext",
        [
            "cytotransform/logicle_ext/FastLogicle.cpp",
            "cytotransform/logicle_ext/Logicle.cpp",
        ],
        include_dirs=["cytotransform/logicle_ext"],
    )
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": dict(build_ext=ExtensionBuilder),
            "zip_safe": False,
        }
    )
