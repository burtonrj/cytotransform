import sys

from .base import Transform
import numpy as np


def _parametrized_log(data: np.ndarray, m: float, t: int) -> np.ndarray:
    """
    Parametrized logarithmic transformation
    """
    return (1/m) * np.log10(data/t) + 1.0


def _inverse_parametrized_log(data: np.ndarray, m: float, t: int) -> np.ndarray:
    """
    Inverse parametrized logarithmic transformation
    """
    return t * (10 ** ((data - 1) * m))


class ParametrizedLogTransform(Transform):
    """
    Parametrized logarithmic transformation
    """
    def __init__(
            self,
            m: float = 4.5,
            t: int = 262144,
            n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        t: int
            The maximum value of the input data that the transformation
            should handle. It sets the upper limit of the linear range
            of the logicle transformation. Values greater than T will
            be transformed, but the transformation may not preserve the
            linearity in the transformed data.
        m: float
            The "magnitude" is the number of decades that the logicle
            transformation spans in logarithmic space. It determines
            the dynamic range of the transformed data. A larger M value
            will result in a greater dynamic range and better separation
            of data points in the transformed space.
        """
        super().__init__(
            transform_function=_inverse_parametrized_log,
            inverse_transform_function=_inverse_parametrized_log,
            parameters={
                't': t,
                'm': m,
            },
            n_jobs=n_jobs
        )
