from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from joblib import Parallel, cpu_count, delayed


class Transform(ABC):
    def __init__(
        self,
        transform_function: Callable,
        inverse_transform_function: Callable,
        parameters: dict,
        n_jobs: int = -1,
    ):
        self._transform_function = transform_function
        self._inverse_transform_function = inverse_transform_function
        self.parameters = parameters
        self.n_jobs: int = n_jobs if n_jobs > 0 else cpu_count()
        self.validation()

    @abstractmethod
    def validation(self):
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self._multiprocess_call(data, self._transform_function)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self._multiprocess_call(data, self._inverse_transform_function)

    def _batches(self, data: np.ndarray) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """
        Split data into N batches, where N is the number of jobs to run in parallel.

        Parameters
        ----------
        data: np.ndarray
            Data to split into batches.

        Returns
        -------
        np.ndarray
            Batches of data.
        """
        n = self.n_jobs if len(data) > 100 else 1
        return np.array_split(data, n)

    def _multiprocess_call(self, data: np.ndarray, func: Callable) -> np.ndarray:
        with Parallel(n_jobs=self.n_jobs) as parallel:
            return np.concatenate(
                parallel(
                    delayed(func)(batch, **self.parameters)
                    for batch in self._batches(data)
                )
            )
