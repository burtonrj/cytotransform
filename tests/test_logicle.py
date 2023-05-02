import sys

from cytotransform.logicle import LogicleTransform
import numpy as np
import pytest

DATA = np.array([-123., -12.5, 0, 0.3, 23.4, 100, 2938, 102939])
TRANSFORMED = np.array([0.09135104, 0.18736521, 0.19995117, 0.20025371, 0.22341899, 0.29181972, 0.60711374, 0.91874843])


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_logicle(n_jobs):
    transformer = LogicleTransform(t=262144, m=4.5, w=0.5, a=0.5, n_jobs=n_jobs)
    if n_jobs == -1:
        assert np.allclose(
            transformer.transform(data=np.concatenate([DATA for _ in range(1000)])),
            np.concatenate([TRANSFORMED for _ in range(1000)])
        )
    assert np.allclose(transformer.transform(data=DATA), TRANSFORMED)


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_logicle_inverse(n_jobs):
    transformer = LogicleTransform(t=262144, m=4.5, w=0.5, a=0.5, n_jobs=n_jobs)
    if n_jobs == -1:
        assert np.allclose(
            transformer.inverse_transform(transformer.transform(data=np.concatenate([DATA for _ in range(1000)]))),
            np.concatenate([DATA for _ in range(1000)])
        )
    assert np.allclose(transformer.inverse_transform(transformer.transform(data=DATA)), DATA)

