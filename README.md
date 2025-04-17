# CUDA Array Addition in Python 🚀

This project demonstrates how to use NVIDIA CUDA with Python to accelerate array computations on the GPU using [Numba](https://numba.pydata.org/) and [NumPy](https://numpy.org/).

## 🚦 What does it do?

It adds two large arrays together using a custom CUDA kernel, leveraging your GPU for massive speedups compared to CPU-only code.

## 🖥️ Requirements

- Python 3.8+
- [Numba](https://numba.pydata.org/)
- [NumPy](https://numpy.org/)
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## ⚡ Setup

```sh
conda install -c nvidia cudatoolkit -y
pip install numba numpy
```

## 🧪 Check your CUDA device

You can verify your CUDA setup with:

```python
import numba
from numba import cuda
print(numba.__version__)
print(cuda.detect())
```

Example output:
```
Found 1 CUDA devices
id 0             b'Tesla T4'                              [SUPPORTED]
                      Compute Capability: 7.5
...
Summary:
        1/1 devices are supported
True
```

## 🚀 Run the demo

```sh
python3 main.py
```

Sample output:
```
Array a  : [0. 1. 2. ... 999997. 999998. 999999.]
Array b  : [0. 1. 2. ... 999997. 999998. 999999.]
a + b    : [0. 2. 4. ... 1999994. 1999996. 1999998.]
```

## 📄 How it works

- [main.py](main.py)  defines a CUDA kernel to add two arrays element-wise.
- Data is transferred to the GPU, processed in parallel, and results are copied back to the CPU.

## 🧠 Learn more

- [Numba CUDA documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

---

Happy GPU computing! 🦾