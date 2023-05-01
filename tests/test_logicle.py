import sys

from cytotransform.logicle import LogicleTransform, FastLogicle
import numpy as np

DATA = np.array([-123., -12.5, 0, 0.3, 23.4, 100, 2938, 102939])
TRANSFORMED = np.array([-123., -12.5, 0, 0.3, 23.4, 100, 2938, 102939])


def test_fastlogicle():
    fl = FastLogicle(T=262144, M=4.5, W=0.5, A=0.0)
    logicle_min, logicle_max = fl.inverse(0.0), fl.inverse(1.0 - sys.float_info.epsilon)
    data = np.clip(DATA, logicle_min, logicle_max)
    scale = np.vectorize(fl.scale)
    assert np.allclose(scale(data), TRANSFORMED)


def test_inverse_logicle():
    pass
