from typing import NamedTuple, Type

import numpy as np
import pandas as pd
import pytest

from cytotransform.asinh import AsinhTransform
from cytotransform.base import Transform
from cytotransform.log import ParametrizedLogTransform
from cytotransform.logicle import LogicleTransform


class TestCase(NamedTuple):
    params: dict
    y: np.ndarray


class TestGroup(NamedTuple):
    klass: Type[Transform]
    cases: list[TestCase]
    x: np.ndarray


AsinhGroup = TestGroup(
    klass=AsinhTransform,
    x=np.array([-10, -5, -1, 0, 0.3, 1, 3, 10, 100, 1000]),
    cases=[
        TestCase(
            params={"t": 1000, "m": 4.0, "a": 1.0},
            y=np.array(
                [
                    -0.200009,
                    -0.139829,
                    -0.000856,
                    0.2,
                    0.303776,
                    0.400856,
                    0.495521,
                    0.600009,
                    0.8,
                    1,
                ]
            ),
        ),
        TestCase(
            params={"t": 1000, "m": 5.0, "a": 0.0},
            y=np.array(
                [
                    -0.6,
                    -0.539794,
                    -0.400009,
                    0,
                    0.295521,
                    0.400009,
                    0.495425,
                    0.6,
                    0.8,
                    1.0,
                ]
            ),
        ),
        TestCase(
            params={"t": 1000, "m": 3.0, "a": 2.0},
            y=np.array(
                [
                    0.199144,
                    0.256923,
                    0.358203,
                    0.4,
                    0.412980,
                    0.441797,
                    0.503776,
                    0.600856,
                    0.800009,
                    1.0,
                ]
            ),
        ),
    ],
)

LogGroup = TestGroup(
    klass=ParametrizedLogTransform,
    x=np.array([0.5, 1, 10, 100, 1000, 1023, 10000, 100000, 262144]),
    cases=[
        TestCase(
            params={"t": 10000, "m": 5.0},
            y=np.array([0.139794, 0.2, 0.4, 0.6, 0.8, 0.801975, 1.0, 1.2, 1.283708]),
        ),
        TestCase(
            params={"t": 1023, "m": 4.5},
            y=np.array(
                [
                    0.264243,
                    0.331139,
                    0.553361,
                    0.775583,
                    0.997805,
                    1.0,
                    1.220028,
                    1.442250,
                    1.535259,
                ]
            ),
        ),
        TestCase(
            params={"t": 262144, "m": 4.5},
            y=np.array(
                [
                    -0.271016,
                    -0.204120,
                    0.018102,
                    0.240324,
                    0.462547,
                    0.464741,
                    0.684768,
                    0.906991,
                    1.0,
                ]
            ),
        ),
    ],
)

LogicleGroup = TestGroup(
    klass=LogicleTransform,
    x=np.array([-10, -5, -1, 0, 0.3, 1, 3, 10, 100, 999]),
    cases=[
        TestCase(
            params={"t": 1000, "w": 1.0, "m": 4.0, "a": 0.0},
            y=np.array(
                [
                    0.067574,
                    0.147986,
                    0.228752,
                    0.25,
                    0.256384,
                    0.271248,
                    0.312897,
                    0.432426,
                    0.739548,
                    0.99988997,
                ]
            ),
        ),
        TestCase(
            params={"t": 1000, "w": 1.0, "m": 4.0, "a": 1.0},
            y=np.array(
                [
                    0.25393797,
                    0.31827791,
                    0.38290087,
                    0.39990234,
                    0.40501019,
                    0.41690381,
                    0.45022804,
                    0.54586672,
                    0.79160413,
                    0.99991194,
                ]
            ),
        ),
    ],
)


@pytest.mark.parametrize(
    "n_jobs,group",
    [
        (1, AsinhGroup),
        (-1, AsinhGroup),
        (1, LogGroup),
        (-1, LogGroup),
        (1, LogicleGroup),
        (-1, LogicleGroup),
    ],
)
def test_transforms_array(n_jobs: int, group: TestGroup):
    for case in group.cases:
        transformer = group.klass(**case.params, n_jobs=n_jobs)
        if n_jobs == -1:
            assert np.allclose(
                transformer.transform(np.concatenate([group.x for _ in range(10000)])),
                np.concatenate([case.y for _ in range(10000)]),
                atol=1e-5,
            )
            assert np.allclose(
                transformer.inverse_transform(
                    np.concatenate([case.y for _ in range(10000)])
                ),
                np.concatenate([group.x for _ in range(10000)]),
                atol=1e-5,
            )

        assert np.allclose(transformer.transform(group.x), case.y, atol=1e-5)
        assert np.allclose(transformer.inverse_transform(case.y), group.x, atol=1e-5)


@pytest.mark.parametrize(
    "n_jobs,group",
    [
        (1, AsinhGroup),
        (-1, AsinhGroup),
        (1, LogGroup),
        (-1, LogGroup),
        (1, LogicleGroup),
        (-1, LogicleGroup),
    ],
)
def test_transforms_dataframe(n_jobs: int, group: TestGroup):
    for case in group.cases:
        transformer = group.klass(**case.params, n_jobs=n_jobs)
        x, y = group.x, case.y
        if n_jobs == -1:
            x, y = np.concatenate([x for _ in range(10000)]), np.concatenate(
                [y for _ in range(10000)]
            )
        df = pd.DataFrame({"x1": x, "x2": x, "x3": x})
        transformed_df = transformer.transform(df)

        for col in transformed_df.columns:
            assert np.allclose(transformed_df[col], y, atol=1e-5)

        inverse_transformed_df = transformer.inverse_transform(transformed_df)
        for col in inverse_transformed_df.columns:
            assert np.allclose(inverse_transformed_df[col], x, atol=1e-5)
