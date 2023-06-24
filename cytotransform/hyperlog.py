from functools import cache

import numpy as np
from scipy.optimize import newton

from .base import Transform


@cache
def intermediates(t_: int, w_: float, m_: float, a_: float):
    w = w_ / (m_ + a_)
    x2 = a_ / (m_ + a_)
    x1 = x2 + w
    x0 = x2 + 2 * w
    b = (m_ + a_) * np.log(10)
    e0 = np.exp(b * x0)
    ca = e0 / w
    fa = np.exp(b * x1) + ca * x1
    a = t_ / (np.exp(b) + ca - fa)
    c = ca * a
    f = fa * a
    return a, b, c, f


def hyperlog(x: np.ndarray, t_: int, w_: float, m_: float, a_: float) -> np.ndarray:
    x = np.asarray(x)
    a, b, c, f = intermediates(t_, w_, m_, a_)

    def _eh(y):
        return a * np.exp(b * y) + c * y - f

    def d_eh(y):
        return a * b * np.exp(b * y) + c

    return np.asarray(newton(_eh, x, fprime=d_eh))


def inverse_hyperlog(x: np.ndarray, t_: int, w_: float, m_: float, a_: float) -> np.ndarray:
    x = np.asarray(x)
    a, b, c, f = intermediates(t_, w_, m_, a_)
    return a * np.exp(b * x) + c * x - f


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
        w: float
            The number of such decades in the approximately linear region or "lower asymptote".
        a: float
             The number of additional decades of negative data values to be included.
        """
        super().__init__(
            transform_function=hyperlog,
            inverse_transform_function=inverse_hyperlog,
            parameters={
                'w_': w,
                't_': t,
                'm_': m,
                'a_': a
            },
            n_jobs=n_jobs
        )

    def validation(self):
        if not self.parameters['t_'] > 0:
            raise ValueError('t must be strictly positive')
        if not self.parameters['m_'] > 0:
            raise ValueError('m must be strictly positive')
        if not 0 < self.parameters['w_'] <= self.parameters['m_']/2:
            raise ValueError('w must be strictly positive and less than or equal to half m')
        if not -self.parameters['w_'] <= self.parameters['a_'] <= (self.parameters['m_'] - 2 * self.parameters['w_']):
            raise ValueError("a must respect the relationship '−W ≤ A ≤ M − 2W'")
