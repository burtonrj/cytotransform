from functools import partial

from .base import Transform
from cytotransform.logicle_ext import FastLogicle
import numpy as np


def logicle(x: np.ndarray, t: int, w: float, m: float, a: float, inverse: bool = False) -> np.ndarray:

    pass


class LogicleTransform(Transform):
    def __init__(
            self,
            w: float = 0.5,
            m: float = 4.5,
            t: int = 262144,
            a: float = 0.0,
            n_jobs: int = -1
    ):
        """
        Logicle transform

        Parameters
        ----------
        x: np.ndarray
            Data to transform.
        t: int
            The maximum value of the input data that the transformation
            should handle. It sets the upper limit of the linear range
            of the logicle transformation. Values greater than T will
            be transformed, but the transformation may not preserve the
            linearity in the transformed data.
        w: float
            The width of the linear range of the transformation. The linear range
            is expressed as a number of decades in logarithmic space, and W
            determines the point where the logicle transformation transitions
            from linear to logarithmic behavior. A larger W value will result
            in a wider linear range and a smoother transition from linear to
            logarithmic scaling.
        m: float
            The "magnitude" is the number of decades that the logicle
            transformation spans in logarithmic space. It determines
            the dynamic range of the transformed data. A larger M value
            will result in a greater dynamic range and better separation
            of data points in the transformed space.
        a: float
            The "asymmetry" is an adjustable parameter that controls
            the position of the linear range relative to the negative
            data values. A larger A value will result in the linear
            range being closer to the negative data values, making it
            easier to visualize and analyze data points close to zero.
            A smaller A value will push the linear range further from
            the negative values, making it easier to visualize and
            analyze data points further from zero.
        inverse: bool
            Whether to perform the inverse transform.
        """
        fastlogicle = FastLogicle(t, w, m, a)
        super().__init__(
            transform_function=fastlogicle.scale,
            inverse_transform_function=fastlogicle.inverse,
            parameters={
                'w': w,
                't': t,
                'm': m,
                'a': a
            },
            n_jobs=n_jobs
        )

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self._multiprocess_call(data, self.transform_function)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self._multiprocess_call(data, self.inverse_transform_function)