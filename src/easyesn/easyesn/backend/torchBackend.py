import sys
import torch
import numpy as np
from . import _BACKEND_CONFIG


__device = torch.device("cpu")
__dtype = torch.float32


def get_device():
    return __device


def set_device(device):
    global __device
    __device = torch.device(device)


def get_dtype():
    return __dtype


def set_dtype(dtype):
    global __dtype
    lut = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    assert dtype in lut.keys()
    __dtype = lut[dtype]


if "device" in _BACKEND_CONFIG:
    set_device(_BACKEND_CONFIG["device"])
    sys.stderr.write("Using device `{0}` as specified by config".format(get_device()))

else:
    if torch.cuda.is_available():
        device = "torch"
        sys.stderr.write("Torch Backend is using CUDA\n")


add = torch.add

substract = torch.sub

dot = lambda x, y: x @ y

multiply = torch.mul

eigenval = torch.eig

eigvals = lambda x: torch.eig(x)[0]

array = lambda x: torch.as_tensor(x, device=__device, dtype=__dtype)

inv = torch.inverse

pinv = torch.pinverse

arctan = torch.atan

vstack = lambda tup: torch.vstack([array(x) for x in tup])

abs = torch.abs

max = torch.max

ones = lambda shape, *args, **kwargs: torch.ones(
    _make_tuple(shape), *args, **kwargs, device=__device, dtype=__dtype
)

zeros = lambda shape, *args, **kwargs: torch.zeros(
    _make_tuple(shape), *args, **kwargs, device=__device, dtype=__dtype
)

empty = lambda shape, *args, **kwargs: torch.empty(
    _make_tuple(shape), *args, **kwargs, device=__device, dtype=__dtype
)

mean = torch.mean

sqrt = torch.sqrt

identity = lambda n: torch.eye(n, device=__device, dtype=__dtype)

rand = torch.rand

power = torch.pow

exp = torch.exp

cosh = torch.cosh

log = torch.log

tanh = torch.tanh

concatenate = torch.cat

sign = torch.sign

argmax = torch.argmax

zeros_like = torch.zeros_like

all = torch.all

var = torch.var

allclose = torch.allclose

# ptp emulation: definition extracted from numpy
ptp = lambda x, axis=None: torch.sub(
    torch.max(x, dim=axis)[0], torch.min(x, dim=axis)[0]
)

count_nonzero = torch.nonzero

nonzero = torch.nonzero

arange = torch.arange

sin = torch.sin

cos = torch.cos

isscalar = np.isscalar

std = torch.std

ceil = torch.ceil

rand = torch.rand

seed = torch.seed

permutation = torch.randperm

randint = lambda *args, **kwargs: torch.randint(*args, **kwargs, device=__device)


def _make_tuple(x):
    if isinstance(x, int):
        return (x,)
    elif isinstance(x, (tuple, list)):
        return x
    else:
        raise ValueError
