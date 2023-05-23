import numpy as np

from .base import Transform


def arcsinh_transform(x: np.ndarray, t: float, m: float, a: float) -> np.ndarray:
    """
    Perform an Arcsinh transformation on the given data.

    Parameters
    ----------
    x: Numpy.Array
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
    x = np.asarray(x)
    pre_scale = np.sinh(m * np.log(10)) / t
    transpose = a * np.log(10)
    divisor = (m + a) * np.log(10)
    return (np.arcsinh(x * pre_scale) + transpose) / divisor


def inverse_arcsinh_transform(
    x: np.ndarray, t: float, m: float, a: float
) -> np.ndarray:
    """
    Perform the inverse Arcsinh transformation on the given transformed data.

    Parameters
    ----------
    x: Numpy.Array
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
    x = np.asarray(x)
    pre_scale = np.sinh(m * np.log(10)) / t
    transpose = a * np.log(10)
    divisor = (m + a) * np.log(10)

    return (np.sinh((x * divisor) - transpose)) / pre_scale


class AsinhTransform(Transform):
    def __init__(
        self, m: float = 4.5, t: int = 262144, a: float = 0.0, n_jobs: int = -1
    ):
        super().__init__(
            transform_function=arcsinh_transform,
            inverse_transform_function=inverse_arcsinh_transform,
            parameters={"t": t, "m": m, "a": a},
            n_jobs=n_jobs,
        )

    def validation(self):
        if not self.parameters["t"] > 0:
            raise ValueError("t must be strictly positive")
        if not self.parameters["m"] > 0:
            raise ValueError("m must be strictly positive")
        if not 0 <= self.parameters["a"] <= self.parameters["m"]:
            raise ValueError("a must respect the relationship '0 ≤ A ≤ M'")
