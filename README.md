# C++Sparse: A C++ Library for Sparse Matrix Computations

This library is a re-write of the original CSparse library by Tim
Davis, within the
[SuiteSparse repo](https://github.com/DrTimothyAldenDavis/SuiteSparse),
following Davis' book "Direct Methods for Sparse Linear Systems" (2006).

My main motivation for creating this library was to simultaneously learn more
about the mechanics of sparse matrix computations, and learn C++.

The original library is written in a terse C style, with many clever memory
optimizations. I have chosen instead to optimize this code for readability and
(self-)education, rather than for performance, though I have tried to keep the
API as similar to that of the original library as possible. I have included
comments where function names have changed, and where I have made significant
changes to the code.

The python tests for this repo rely on my
[suitesparseget_py](https://github.com/broesler/suitesparseget_py). This package
provides a Python interface to the SuiteSparse Matrix Collection, so that you
can load into `scipy.sparse` matrices. It mimics the behavior of the `ssget`
package built into the original CSparse library for use with MATLAB.

## Installation

### Prerequisites
* **C++ Compiler** with C++23 suppor (*e.g.* `clang++` or `g++`)
* **CMake** (>= 3.25)
* **[uv](https://astral.sh/uv/)** (Python package and environment manager)

### Quickstart
Clone the repository:

```bash
gh repo clone broesler/CppSparse C++Sparse
cd C++Sparse
```

First, sync the environment with the `uv` package manager:

```bash
uv sync --extra dev
```

Then, build and test the C++ library:

```bash
make tests
./build/Release/test_csparse
```

Then, build and test the Python interface:

```bash
uv run pytest
```

The python tests roughly replicate the tests in the original CSparse library
(under `CSparse/MATLAB/Test`), and also include some additional tests for
the Python interface.

To make the figures in the python tests (saved to the `test_figures/` directory),

```bash
uv run pytest --make-figures
```

**Warning:** Making the figures is slow. We recommend that you run the tests for
individual modules or classes one at a time, *e.g.*

```bash
uv run pytest --make-figures -k 'test_amd'
```

Note that figure windows may not be visible while the tests are running, but the
files will still be saved.

## Demos
In addition to the unit tests, there are also demo and example scripts.

To run the C++ demos:

```bash
make run_demos
```

To run the python demos:

```bash
uv run python python/demo/demo1.py ./data/t1
uv run python python/demo/demo2.py
uv run python python/demo/demo3.py
```

Note that depending on your Matplotlib backend, you may need to close the figure
window to continue on to the next plot for the python demos.

Additional python scripts are located in the `python/scripts` directory. Many of
them follow experiments and exercises as presented in Davis' book.

## Usage
The library is intended to be run through the python interface, although the C++
functions are available for use. The python interface is a thin wrapper around
the C++ interface, analagous to the MATLAB interface of the original CSparse
library. The python interface is designed to work seamlessly with `scipy.sparse`
matrices, so you can use it to solve sparse linear systems, compute matrix
inverses, and perform other sparse matrix computations.
