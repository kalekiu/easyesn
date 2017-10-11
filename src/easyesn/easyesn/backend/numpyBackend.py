import numpy as np
from .backendBase import *

class NumpyBackend(BackendBase):
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