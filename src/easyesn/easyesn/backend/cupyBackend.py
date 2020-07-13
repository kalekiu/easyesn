import cupy as cp

def add(x, y):
	return cp.add(x, y)

def substract(x, y):
	return cp.substract(x, y)

def dot(x, y):
	return cp.dot(x, y)

def multiply(x, y):
	return cp.multiply(x, y)

def eigenval(x):
	import numpy as np
	np_x = cp.asnumpy(x)
	results = np.linalg.eig(np_x)
	return cp.array(results[0]), cp.array([results[1]])

def array(x):
	return cp.array(x)

def inv(x):
	return cp.linalg.inv(x)

def pinv(x):
	return cp.linalg.pinv(x)

def arctan(x):
	return cp.arctan(x)

def vstack(x):
	return cp.vstack(x)

def abs(x):
	return cp.abs(x)

def max(x):
	return cp.max(x)

def ones(x):
	return cp.ones(x)

def zeros(x):
	return cp.zeros(x)

def empty(x):
	return cp.empty(x)

def mean(x, axis=None):
	return cp.mean(x, axis)

def sqrt(x):
	return cp.sqrt(x)

def identity(x):
	return cp.identity(x)

def rand(*x):
	return cp.random.rand(*x)

def power(x, y):
	return cp.power(x, y)

def exp(x):
	return cp.exp(x)

def cosh(x):
    return cp.cosh(x)  

def log(x):
    return cp.log(x)

def tanh(x):
	return cp.tanh(x)

def concatenate(tuple, axis=0):
	return cp.concatenate(tuple, axis=axis)

def sign(x):
	return cp.sign(x)

def argmax(x, axis):
    return cp.argmax(x, axis)

def zeros_like(x):
    return cp.zeros_like(x)

def ptp(x, axis=None):
    return cp.subtract(cp.amax(x, axis), cp.amin(x, axis))
