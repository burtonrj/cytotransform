from .base import Transform
import numpy as np


def _arcsinh_transform(data: np.ndarray, t: float, m: float, a: float) -> np.ndarray:
    """
    Perform an Arcsinh transformation on the given data.

    Parameters
    ----------
    data: Numpy.Array
        The input data (e.g., Mass Cytometry or flow cytometry data).
    t: float
        Parameter specifying the top of the scale
    m: float
        Parameter for the number of decades
    a: float
        Parameter for the number of additional negative decades

    Returns
    -------
    Numpy.Array
        The Arcsinh-transformed data.
    """
    pre_scale = np.sinh(m * np.log(10)) / t
    transpose = a * np.log(10)
    divisor = (m + a) * np.log(10)
    return (np.arcsinh(data * pre_scale) + transpose) / divisor


def _inverse_arcsinh_transform(data: np.ndarray, t: float, m: float, a: float) -> np.ndarray:
    """
    Perform the inverse Arcsinh transformation on the given transformed data.

    Parameters
    ----------
    data: Numpy.Array
        The transformed data
    t: float
        Parameter specifying the top of the scale
    m: float
        Parameter for the number of decades
    a: float
        Parameter for the number of additional negative decades

    Returns
    -------
    Numpy.Array
        The original (untransformed) data.
    """
    pre_scale = np.sinh(m * np.log(10)) / t
    transpose = a * np.log(10)
    divisor = (m + a) * np.log(10)

    return (np.sinh((data * divisor) - transpose)) / pre_scale


class AsinhTransform(Transform):

    def __init__(self, cofactor: float = 150.0, n_jobs: int = -1):
        super().__init__(
            transform_function=_arcsinh_transform,
            inverse_transform_function=_inverse_arcsinh_transform,
            parameters={"cofactor": cofactor},
            n_jobs=n_jobs
        )
