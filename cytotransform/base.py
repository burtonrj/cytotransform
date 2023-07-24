from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
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

    def transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return self._multiprocess_call_df(data, self._transform_function)
        return self._multiprocess_call_array(data, self._transform_function)

    def inverse_transform(
        self, data: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return self._multiprocess_call_df(data, self._inverse_transform_function)
        return self._multiprocess_call_array(data, self._inverse_transform_function)

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
        n = self.n_jobs if len(data) > 10000 else 1
        return np.array_split(data, n)

    def _multiprocess_call_array(self, data: np.ndarray, func: Callable) -> np.ndarray:
        if self.n_jobs in [0, 1]:
            return func(data, **self.parameters)
        with Parallel(n_jobs=self.n_jobs) as parallel:
            return np.concatenate(
                parallel(
                    delayed(func)(batch, **self.parameters)
                    for batch in self._batches(data)
                )
            )

    def _multiprocess_call_df(self, data: pd.DataFrame, func: Callable) -> pd.DataFrame:
        if self.n_jobs in [0, 1]:
            return pd.concat(
                [
                    pd.Series(
                        func(data[col], **self.parameters), name=col, index=data.index
                    )
                    for col in data.columns
                ],
                axis=1,
            )
        with Parallel(n_jobs=self.n_jobs) as parallel:
            transformed = parallel(
                delayed(func)(data[col], **self.parameters) for col in data.columns
            )
            return pd.concat(
                [
                    pd.Series(t, name=col, index=data.index)
                    for t, col in zip(transformed, data.columns)
                ],
                axis=1,
            )
