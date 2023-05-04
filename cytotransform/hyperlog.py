import sys

from .base import Transform
from scipy.optimize import root_scalar
import numpy as np


def _eh(y, t, w, m, a):
    a = (a - w) / (4 * (t - w / 2)) ** (1 / m)
    b = np.log((t - w / 2) / a)
    c = (t - w / 2) - a * np.exp(b * m)
    f = a * np.exp(b * (m + 1))
    return a, b, c, f


def eh(y, t, w, m, a):
    a, b, c, f = _eh(y, t, w, m, a)
    return a * np.exp(b * y) + c * y - f


def eh_inverse(x, t, w, m, a):
    a, b, c, f = _eh(x, t, w, m, a)
    return (x + f - c * m) / (a * np.exp(b * m) + c)


def _hyperlog(x, t, w, m, a):
    def objective(y):
        return eh(y, t, w, m, a) - x

    result = root_scalar(objective, bracket=[-a, t])
    return result.root


def _hyperlog_inverse(x, t, w, m, a):
    def objective(y):
        return eh_inverse(y, t, w, m, a) - x

    result = root_scalar(objective, bracket=[-a, t])
    return result.root


class HyperlogTransform(Transform):
    """

    """
    def __init__(
            self,
            w: float = 1.0,
            m: float = 4.5,
            t: int = 262144,
            a: float = 0.0,
            n_jobs: int = -1
    ):
        """
        Parameters
        ----------
        w: float
            Width of the linear region
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
        a: float
            Lower asymptote
        """
        super().__init__(
            transform_function=_hyperlog,
            inverse_transform_function=_hyperlog_inverse,
            parameters={
                't': t,
                'm': m,
            },
            n_jobs=n_jobs
        )
