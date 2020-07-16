import cupy as cp
import numpy

add = cp.add

substract = cp.subtract

dot = cp.dot

multiply = cp.multiply

eigenval = np.linalg.eig

array = cp.array

inv = cp.linalg.inv

pinv = cp.linalg.pinv

arctan = cp.arctan

vstack = cp.vstack

abs = cp.abs

max = cp.max

ones = cp.ones

zeros = cp.zeros

empty = cp.empty

mean = cp.mean

sqrt = cp.sqrt

identity = cp.identity

rand = cp.random.rand

power = cp.power

exp = cp.exp

cosh = cp.cosh

log = cp.log

tanh = cp.tanh

concatenate = cp.concatenate

sign = cp.sign

argmax = cp.argmax

zeros_like = cp.zeros_like

all = cp.all

var = cp.var

allclose = cp.allclose

# ptp emulation: definition extracted from numpy
ptp = lambda x, axis=None: cp.subtract(cp.amax(x, axis), cp.amin(x, axis))

count_nonzero = cp.count_nonzero

arange = np.arange

sin = cp.sin

cos = cp.cos

isscalar = cp.isscalar

std = cp.std

ceil = cp.ceil

rand = cp.random.rand

seed = cp.random.seed

permutation = cp.random.permutation

randint = cp.random.randint

random_integers = cp.random.random_integers
