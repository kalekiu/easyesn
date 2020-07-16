import numpy as np
from numpy import random
import scipy as sp
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

add = np.add

substract = np.subtract

dot = np.dot

multiply = np.multiply

eigenval = np.linalg.eig

def eigvals(x):
    try:
        #this is just a good approximation, so this code might fail
        _eig = sp.sparse.linalg.eigs(x)[0]
    except ArpackNoConvergence:
        #this is the safe fall back method to calculate the EV
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