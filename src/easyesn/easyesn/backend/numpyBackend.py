import numpy as np

def add(x, y):
	return np.add(x, y)

def substract(x, y):
	return np.substract(x, y)

def dot(x, y):
	return np.dot(x, y)

def multiply(x, y):
	return np.multiply(x, y)

def eigenval(x):
	return np.linalg.eig(x)

def array(x):
	return np.array(x)

def inv(x):
	return np.linalg.inv(x)

def pinv(x):
	return np.linalg.pinv(x)

def arctan(x):
	return np.arctan(x)

def vstack(x):
	return np.vstack(x)

def abs(x):
	return np.abs(x)

def max(x):
	return np.max(x)

def ones(x):
	return np.ones(x)

def zeros(x):
	return np.zeros(x)

def empty(x):
	return np.empty(x)

def mean(x):
	return np.mean(x)

def sqrt(x):
	return np.sqrt(x)

def identity(x):
	return np.identity(x)

def rand(*x):
	return np.random.rand(*x)

def power(x, y):
	return np.power(x, y)

def exp(x):
	return np.exp(x)

def tanh(x):
	return np.tanh(x)

def concatenate(tuple, axis=0):
	return np.concatenate(tuple, axis=axis)

def sign(x):
	return np.sign(x)
