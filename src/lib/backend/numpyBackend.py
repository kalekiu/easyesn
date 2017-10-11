import numpy as np
from backendBase import *

class NumpyBackend(backendBase):
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
