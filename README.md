# Cytotransfrom

![PyPI](https://img.shields.io/pypi/v/cytotransform)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cytotransform)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/cytotransform)
![PyPI - License](https://img.shields.io/pypi/l/cytotransform)
![Codecov](https://img.shields.io/codecov/c/github/burtonrj/cytotransform)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/burtonrj/cytotransform/build.yaml)

## Description

Cytotransform is a python package for transforming flow cytometry data. It implements the following transformations
according to the GatingML 2.0 definitions (https://flowcyt.sourceforge.net/gating/latest.pdf):

- Parametrized logarithmic transformation
- Parametrized inverse hyperbolic sine transformation (asinh)
- Logicle transformation
- Hyperlog transformation

Each transformation is implemented as a `Transform` class with a `transform` method that takes a numpy array as input
and returns the transformed array. The `Transform` class also has a `transform_inverse` method that takes a numpy array
as input and returns the inverse transformed array. Each implementation includes validation of the input parameters. The
transform classes support multiprocessing out of the box and if `n_jobs` is set to more than 0, then the input data will
be split into `n` batches depending on the number of cores available and each batch will be transformed in parallel.
If `n_jobs` is set to -1, then all available cores will be used. If `n_jobs` is set to 0, then no multiprocessing will
be used.

Cytotransform is thanks to the fantastic community of scientists and developers in the single cell and flow
cytometry data analysis ecosystem. It implements the FastLogicle C++ library for logicle transformations
originally implemented by Wayne A Moore and David R Parks (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4761345/).
Code was inspired by the great work by Scott White (https://github.com/whitews/FlowKit) and Brian Teague
(https://github.com/cytoflow).

## Installation

Cytotransform can be installed from PyPi using pip:

```bash
pip install cytotransform
```

## Usage

### Parametrized logarithmic transformation

```python
from cytotransform import ParametrizedLogTransform
transformer = ParametrizedLogTransform(m=4.5, t=262144, n_jobs=-1)
transformed_data = transformer.transform(data)
data = transformer.inverse_transform(transformed_data)
```

### Parametrized inverse hyperbolic sine transformation (asinh)

```python
from cytotransform import AsinhTransform
transformer = AsinhTransform(m=4.5, t=262144, a=0.0, n_jobs=-1)
transformed_data = transformer.transform(data)
data = transformer.inverse_transform(transformed_data)
```

### Logicle transformation

```python
from cytotransform import LogicleTransform
transformer = LogicleTransform(t=262144, w=0.5, m=4.5, a=0.0, n_jobs=-1)
transformed_data = transformer.transform(data)
data = transformer.inverse_transform(transformed_data)
```

### Hyperlog transformation

```python
from cytotransform import HyperlogTransform
transformer = HyperlogTransform(t=262144, w=0.5, m=4.5, a=0.0, n_jobs=-1)
transformed_data = transformer.transform(data)
data = transformer.inverse_transform(transformed_data)
```

## License

Cytotransform is licensed under the MIT license, is free to use, and comes with no warranty whatsoever.
