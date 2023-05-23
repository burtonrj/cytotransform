import numpy as np

from .base import Transform


def parametrized_log(x: np.ndarray, m: float, t: int) -> np.ndarray:
    """
    Parametrized logarithmic transformation
    """
    x = np.asarray(x)
    return (1 / m) * np.log10(x / t) + 1.0


def inverse_parametrized_log(x: np.ndarray, m: float, t: int) -> np.ndarray:
    """
    Inverse parametrized logarithmic transformation
    """
    x = np.asarray(x)
    return t * (10 ** ((x - 1) * m))


class ParametrizedLogTransform(Transform):
    """
    Parametrized logarithmic transformation
    """

    def __init__(self, m: float = 4.5, t: int = 262144, n_jobs: int = -1):
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
            transform_function=parametrized_log,
            inverse_transform_function=inverse_parametrized_log,
            parameters={
                "t": t,
                "m": m,
            },
            n_jobs=n_jobs,
        )

    def validation(self):
        if not self.parameters["t"] > 0:
            raise ValueError("t must be strictly positive")
        if not self.parameters["m"] > 0:
            raise ValueError("m must be strictly positive")
