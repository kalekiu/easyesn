"""
    Basic implementation of an ESN.
"""

import numpy as np
import numpy.random as rnd
import dill as pickle
import scipy as sp

class BaseESN(object):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                 leak_rate=1.0, sparseness=0.2, random_seed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weight_generation='naive', bias=1.0, output_bias=1.0, output_input_scaling=1.0,
                 feedback=False, scale_input_matrix=False, input_density=1.0):

        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output

        self.spectral_radius = spectral_radius
        self.noise_level = noise_level
        self.sparseness = sparseness
        self.leak_rate = leak_rate
        self.input_density = input_density

        if input_scaling is None:
            input_scaling = np.ones(n_input)
        if np.isscalar(input_scaling):
            input_scaling = np.repeat(input_scaling, n_input)
        else:
            if len(input_scaling) != self.n_input:
                raise ValueError("Dimension of input_scaling ({0}) does not match the input data dimension ({1})".format(len(input_scaling), n_input))

        self._input_scaling_matrix = np.diag(input_scaling)
        self._expanded_input_scaling_matrix = np.diag(np.vstack((1.0, input_scaling.reshape(-1, 1))).flatten())

        self.out_activation = out_activation
        self.out_inverse_activation = out_inverse_activation

        if random_seed is not None:
            rnd.seed(random_seed)

        self.bias = bias
        self.output_bias = output_bias
        self.output_input_scaling = output_input_scaling
        self._create_reservoir(weight_generation, feedback)

    """
        Generates a random rotation matrix, used in the SORM initilization (see http://ftp.math.uni-rostock.de/pub/preprint/2012/pre12_01.pdf)
    """
    def create_random_rotation_matrix(self):
        h = rnd.randint(low=0, high=self.n_reservoir)
        k = rnd.randint(low=0, high=self.n_reservoir)

        phi = rnd.rand(1)*2*np.pi

        Q = np.identity(self.n_reservoir)
        Q[h, h] = np.cos(phi)
        Q[k, k] = np.cos(phi)

        Q[h, k] = -np.sin(phi)
        Q[k, h] = np.sin(phi)

        return Q

    """
        Internal method to create the matrices W_in, W and W_fb of the ESN
    """
    def _create_reservoir(self, weight_generation, feedback=False, verbose=False):
        #naive generation of the matrix W by using random weights
        if weight_generation == 'naive':
            #random weight matrix from -0.5 to 0.5
            self._W = rnd.rand(self.n_reservoir, self.n_reservoir) - 0.5

            #set sparseness% to zero
            mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparseness
            self._W[mask] = 0.0

            _W_eigenvalues = np.abs(np.linalg.eig(self._W)[0])
            self._W *= self.spectral_radius / np.max(_W_eigenvalues)

        #generation using the SORM technique (see http://ftp.math.uni-rostock.de/pub/preprint/2012/pre12_01.pdf)
        elif weight_generation == "SORM":
            self._W = np.identity(self.n_reservoir)

            number_nonzero_elements = self.sparseness * self.n_reservoir * self.n_reservoir
            i = 0

            while np.count_nonzero(self._W) < number_nonzero_elements:
                i += 1
                Q = self.create_random_rotation_matrix()
                self._W = Q.dot(self._W)
            print(i)
            self._W *= self.spectral_radius

        #generation using the proposed method of Yildiz
        elif weight_generation == 'advanced':
            #two create W we must follow some steps:
            #at first, create a W = |W|
            #make it sparse
            #then scale its spectral radius to rho(W) = 1 (according to Yildiz with x(n+1) = (1-a)*x(n)+a*f(...))
            #then change randomly the signs of the matrix

            #random weight matrix from 0 to 0.5

            self._W = rnd.rand(self.n_reservoir, self.n_reservoir) / 2

            #set sparseness% to zero
            mask = rnd.rand(self.n_reservoir, self.n_reservoir) > self.sparseness
            self._W[mask] = 0.0

            from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
            #just calculate the largest EV - hopefully this is the right code to do so...
            try:
                #this is just a good approximation, so this code might fail
                _W_eigenvalue = np.max(np.abs(sp.sparse.linalg.eigs(self._W, k=1)[0]))
            except ArpackNoConvergence:
                #this is the safe fall back method to calculate the EV
                _W_eigenvalue = np.max(np.abs(sp.linalg.eigvals(self._W)))
            #_W_eigenvalue = np.max(np.abs(np.linalg.eig(self._W)[0]))

            self._W *= self.spectral_radius / _W_eigenvalue

            if verbose:
                M = self.leak_rate*self._W + (1 - self.leak_rate)*np.identity(n=self._W.shape[0])
                M_eigenvalue = np.max(np.abs(np.linalg.eig(M)[0]))#np.max(np.abs(sp.sparse.linalg.eigs(M, k=1)[0]))
                print("eff. spectral radius: {0}".format(M_eigenvalue))

            #change random signs
            random_signs = np.power(-1, rnd.random_integers(self.n_reservoir, self.n_reservoir))

            self._W = np.multiply(self._W, random_signs)
        elif weight_generation == 'custom':
            pass
        else:
            raise ValueError("The weight_generation property must be one of the following values: naive, advanced, SORM, custom")

        #check of the user is really using one of the internal methods, or wants to create W by his own
        if (weight_generation != 'custom'):
            #random weight matrix for the input from -0.5 to 0.5
            self._W_input = np.random.rand(self.n_reservoir, 1+self.n_input)-0.5

            #scale the input_density to prevent saturated reservoir nodes
            if (self.input_density != 1.0):
                #make the input matrix as dense as requested
                input_topology = (np.ones_like(self._W_input) == 1.0)
                nb_non_zero_input = int(self.input_density * self.n_input)
                for n in range(self.n_reservoir):
                    input_topology[n][rnd.permutation(np.arange(1+self.n_input))[:nb_non_zero_input]] = False

                self._W_input[input_topology] = 0.0

            self._W_input = self._W_input.dot(self._expanded_input_scaling_matrix)

        #create the optional feedback matrix
        if feedback:
            self._W_feedback = np.random.rand(self.n_reservoir, 1+self.n_output) - 0.5

    """
        Updates the inner states. Returns the UNSCALED but reshaped input of this step.
    """
    def update(self, inputData):
        #reshape the data
        u = inputData.reshape(self.n_input, 1)

        #update the states
        self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x) +
                                                                          (np.random.rand()-0.5)*self.noise_level)
        self._x = np.asarray(self._x)

        return u

    """
        Updates the inner states. Returns the UNSCALED but reshaped input of this step.
    """
    def update_feedback(self, inputData, outputData):
        #the input is allowed to be "empty" (size=0)
        if self.n_input != 0:
            #reshape the data
            u = inputData.reshape(self.n_input, 1)
            outputData = outputData.reshape(self.n_output, 1)

            #update the states
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W_input, np.vstack((self.bias, u))) + np.dot(self._W, self._x) +
                                                                              np.dot(self._W_feedback, np.vstack((self.output_bias, outputData))) + (np.random.rand()-0.5)*self.noise_level)

            return u
        else:
            #reshape the data
            outputData = outputData.reshape(self.n_output, 1)
            #update the states
            self._x = (1.0-self.leak_rate)*self._x + self.leak_rate*np.arctan(np.dot(self._W, self._x) + np.dot(self._W_feedback, np.vstack((self.output_bias, outputData))) +
                                                                              (np.random.rand()-0.5)*self.noise_level)

            return np.empty((0, 1))

    """
        Saves the ESN by pickling it.
    """
    def save(self, path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    """
        Loads a previously pickled ESN.
    """
    def load(path):
        f = open(path, "rb")
        result = pickle.load(f)
        f.close()
        return result
