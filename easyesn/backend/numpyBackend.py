import numpy as np
from numpy import random
import scipy as sp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

add = np.add

substract = np.subtract

dot = np.dot

multiply = np.multiply

eigenval = np.linalg.eig


def ishermitian(A, tol=1e-6):
    x = sp.rand(A.shape[0], 1)
    y = sp.rand(A.shape[0], 1)
    if A.dtype == complex:
        x = x + 1.0j * sp.rand(A.shape[0], 1)
        y = y + 1.0j * sp.rand(A.shape[0], 1)
    xAy = np.dot((A * x).conjugate().T, y)
    xAty = np.dot(x.conjugate().T, A * y)
    diff = float(np.abs(xAy - xAty) / np.sqrt(np.abs(xAy * xAty)))
    if diff < tol:
        diff = 0
        return True
    else:
        return False


def eigvals(x):
    try:
        if sp.sparse.isspmatrix(x):
            if ishermitian(A):
                _eig = sp.sparse.linalg.eigsh(x)[0]
            else:
                _eig = sp.sparse.linalg.eigs(x)[0]
        else:
            A = np.asmatrix(x)
            if ishermitian(A):
                _eig = sp.linalg.eigvalsh(x)
            else:
                _eig = sp.linalg.eigvals(x)
    except ArpackNoConvergence:
        _eig = sp.linalg.eigvals(x)
    return _eig


array = np.array

inv = np.linalg.inv

pinv = np.linalg.pinv

arctan = np.arctan

vstack = np.vstack

abs = np.abs

max = np.max

ones = np.ones

zeros = np.zeros

empty = np.empty

mean = np.mean

sqrt = np.sqrt

identity = np.identity

rand = np.random.rand

power = np.power

exp = np.exp

cosh = np.cosh

log = np.log

tanh = np.tanh

concatenate = np.concatenate

sign = np.sign

argmax = np.argmax

zeros_like = np.zeros_like

all = np.all

correlate = np.correlate

var = np.var

allclose = np.allclose

ptp = np.ptp

randint = random.randint
