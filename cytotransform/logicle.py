import sys

import numpy as np

from logicle_ext import FastLogicle  # type: ignore

from .base import Transform


def fastlogicle_wrapper(
    x: np.ndarray, t: int, w: float, m: float, a: float
) -> np.ndarray:
    fl = FastLogicle(T=t, W=w, M=m, A=a)
    logicle_min, logicle_max = fl.inverse(0.0), fl.inverse(1.0 - sys.float_info.epsilon)
    x = np.clip(x, logicle_min, logicle_max)
    return np.vectorize(fl.scale)(x)


def fastlogicle_inverse_wrapper(
    x: np.ndarray, t: int, w: float, m: float, a: float
) -> np.ndarray:
    fl = FastLogicle(T=t, W=w, M=m, A=a)
    return np.vectorize(fl.inverse)(x)


class LogicleTransform(Transform):
    """
    Logicle transform, implemented using the FastLogicle class as originally published in the following paper:

    Moore WA, Parks DR. Update for the logicle data scale including operational code implementations. Cytometry A. 2012
    Apr;81(4):273-7. doi: 10.1002/cyto.a.22030. Epub 2012 Mar 12. PMID: 22411901; PMCID: PMC4761345.
    """

    def __init__(
        self,
        w: float = 0.5,
        m: float = 4.5,
        t: int = 262144,
        a: float = 0.0,
        n_jobs: int = -1,
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
        """
        self._fl = FastLogicle(T=t, M=m, W=w, A=a)
        super().__init__(
            transform_function=fastlogicle_wrapper,
            inverse_transform_function=fastlogicle_inverse_wrapper,
            parameters={"w": w, "t": t, "m": m, "a": a},
            n_jobs=n_jobs,
        )

    def validation(self):
        if not self.parameters["t"] > 0:
            raise ValueError("t must be strictly positive")
        if not self.parameters["m"] > 0:
            raise ValueError("m must be strictly positive")
        if not 0 <= self.parameters["w"] <= self.parameters["m"] / 2:
            raise ValueError(
                "w must be strictly positive and less than or equal to half m"
            )
        if (
            not -self.parameters["w"]
            <= self.parameters["a"]
            <= (self.parameters["m"] - 2 * self.parameters["w"])
        ):
            raise ValueError("a must respect the relationship '−W ≤ A ≤ M − 2W'")
