import sys
import torch
import numpy as np

if torch.cuda.is_available():
    sys.stderr.write('Torch Backend is using CUDA\n')
else:
    sys.stderr.write('Torch Backend is not using CUDA\n')

add = torch.add

substract = torch.sub

dot = lambda x, y: x@y

multiply = torch.mul

eigenval = torch.eig

eigvals = lambda x: torch.eig(x)[0]

array = torch.as_tensor

inv = torch.inverse

pinv = torch.pinverse

arctan = torch.atan

#atleast_2d needed by vstack emulation
def atleast_2d(*arys):
    res = []
    for ary in arys:
        ary = torch.as_tensor(ary, dtype=torch.double)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[np.newaxis, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

#definition of vstack emulation taken from numpy
def vstack(tup):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    x = torch.cat(arrs, dim=0)
    return x

abs = torch.abs

max = torch.max

ones = torch.ones

zeros = torch.zeros

empty = torch.empty

mean = torch.mean

sqrt = torch.sqrt

identity = torch.nn.Identity

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
ptp = lambda x, axis=None: torch.sub(torch.max(x, dim=axis)[0], torch.min(x, dim=axis)[0])

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

randint = torch.randint