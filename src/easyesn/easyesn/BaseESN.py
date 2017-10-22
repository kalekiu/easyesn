"""
    Basic implementation of an ESN.
"""

#from __future__ import absolute_import

import numpy as np
import numpy.random as rnd
import dill as pickle
import scipy as sp
import progressbar

#import backend as B

from . import backend as B

class BaseESN(object):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectralRadius=1.0, noiseLevel=0.01, inputScaling=None,
                 leakingRate=1.0, feedbackScaling = 1.0, sparseness=0.2, random_seed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weight_generation='naive', bias=1.0, output_bias=1.0, outputInputScaling=1.0,
                 feedback=False, scale_input_matrix=False, input_density=1.0, activation = B.tanh):

        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output

        self._spectralRadius = spectralRadius
        self.noise_level = noiseLevel
        self.sparseness = sparseness
        self._leakingRate = leakingRate
        self._feedbackScaling = feedbackScaling
        self.input_density = input_density
        self._activation = activation

        if inputScaling is None:
            self._inputScaling = 1.0
        if np.isscalar(self._inputScaling):
            inputScaling = B.ones(n_input) * self._inputScaling
        else:
            if len(self._inputScaling) != self.n_input:
                raise ValueError("Dimension of inputScaling ({0}) does not match the input data dimension ({1})".format(len(self._inputScaling), n_input))
            self._inputScaling = inputScaling

        self._expanded_inputScaling = B.vstack((1.0, inputScaling.reshape(-1,1))).flatten()

        self.out_activation = out_activation
        self.out_inverse_activation = out_inverse_activation

        if random_seed is not None:
            rnd.seed(random_seed)

        self.bias = bias
        self.output_bias = output_bias
        self.outputInputScaling = outputInputScaling
        self._create_reservoir(weight_generation, feedback)


    def setSpectralRadius(self, newSpectralRadius):
        self._W = self._W * ( newSpectralRadius / self._spectralRadius )
        self._spectralRadius = newSpectralRadius
        #TODO numerical instability

    def setLeakingRate(self, newLeakingRate):
        self._leakingRate = newLeakingRate

    def setInputScaling(self, newInputScaling):
        inputScaling = B.ones(self.n_input) * self._inputScaling
        self._expanded_inputScaling = B.vstack((1.0, inputScaling.reshape(-1, 1))).flatten()
        self._W_input = self._W_input * ( self._expanded_inputScaling / self._inputScaling )
        self._inputScaling = newInputScaling

    def setFeedbackScaling(self, newFeedbackScaling):
        self._W_feedback = self._W_feedback * ( newFeedbackScaling / self._feedbackScaling)
        self._feedbackScaling = newFeedbackScaling


    def resetState(self):
        self._x = B.zeros_like(self._x)

    def propagate(self, inputData, transientTime, verbose=0, x=None ):
        if x is None:
            x = self._x

        trainLength = len(inputData)

        # define states' matrix
        X = B.zeros((1 + self.n_input + self.n_reservoir, trainLength - transientTime))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(trainLength):
            u = self.update(inputData[t], x=x)
            if (t >= transientTime):
                #add valueset to the states' matrix
                X[:,t-transientTime] = B.vstack((self.output_bias, self.outputInputScaling*u, x))[:,0]
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        return X

    """
        Generates a random rotation matrix, used in the SORM initilization (see http://ftp.math.uni-rostock.de/pub/preprint/2012/pre12_01.pdf)
    """
    def create_random_rotation_matrix(self):
        h = rnd.randint(low=0, high=self.n_reservoir)
        k = rnd.randint(low=0, high=self.n_reservoir)

        phi = rnd.rand(1)*2*np.pi

        Q = B.identity(self.n_reservoir)
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

            _W_eigenvalues = B.abs(np.linalg.eig(self._W)[0])
            self._W *= self._spectralRadius / B.max(_W_eigenvalues)

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
            self._W *= self._spectralRadius

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
            mask = B.rand(self.n_reservoir, self.n_reservoir) > self.sparseness
            self._W[mask] = 0.0

            from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
            #just calculate the largest EV - hopefully this is the right code to do so...
            try:
                #this is just a good approximation, so this code might fail
                _W_eigenvalue = B.max(np.abs(sp.sparse.linalg.eigs(self._W, k=1)[0]))
            except ArpackNoConvergence:
                #this is the safe fall back method to calculate the EV
                _W_eigenvalue = B.max(B.abs(sp.linalg.eigvals(self._W)))
            #_W_eigenvalue = B.max(B.abs(np.linalg.eig(self._W)[0]))

            self._W *= self._spectralRadius / _W_eigenvalue

            if verbose:
                M = self._leakingRate*self._W + (1 - self._leakingRate)*np.identity(n=self._W.shape[0])
                M_eigenvalue = B.max(B.abs(np.linalg.eig(M)[0]))#np.max(np.abs(sp.sparse.linalg.eigs(M, k=1)[0]))
                print("eff. spectral radius: {0}".format(M_eigenvalue))

            #change random signs
            random_signs = B.power(-1, rnd.random_integers(self.n_reservoir, self.n_reservoir))

            self._W = B.multiply(self._W, random_signs)
        elif weight_generation == 'custom':
            pass
        else:
            raise ValueError("The weight_generation property must be one of the following values: naive, advanced, SORM, custom")

        #check of the user is really using one of the internal methods, or wants to create W by his own
        if (weight_generation != 'custom'):
            #random weight matrix for the input from -0.5 to 0.5
            self._W_input = B.rand(self.n_reservoir, 1 + self.n_input)-0.5

            #scale the input_density to prevent saturated reservoir nodes
            if (self.input_density != 1.0):
                #make the input matrix as dense as requested
                input_topology = (np.ones_like(self._W_input) == 1.0)
                nb_non_zero_input = int(self.input_density * self.n_input)
                for n in range(self.n_reservoir):
                    input_topology[n][rnd.permutation(np.arange(1+self.n_input))[:nb_non_zero_input]] = False

                self._W_input[input_topology] = 0.0

            self._W_input = self._W_input * self._expanded_inputScaling

        #create the optional feedback matrix
        if feedback:
            self._W_feedback = B.rand(self.n_reservoir, 1 + self.n_output) - 0.5
            self._W_feedback *= self._feedbackScaling
        else:
            self._W_feedback = None


    def calculateLinearNetworkTransmissions(self, u, x=None):
        if x is None:
            x = self._x

        return B.dot(self._W_input, B.vstack((self.bias, u))) + B.dot(self._W, x)

    """
        Updates the inner states. Returns the UNSCALED but reshaped input of this step.
    """
    def update(self, inputData, outputData=None, x=None):
        if x is None:
            x = self._x

        if self._W_feedback is None:
            #reshape the data
            u = inputData.reshape(self.n_input, 1)

            #update the states
            transmission = self.calculateLinearNetworkTransmissions(u, x)
            x *= (1.0-self._leakingRate)
            x += self._leakingRate * self._activation(transmission + (B.rand()-0.5)*self.noise_level)
        
            return u

        else:
            #the input is allowed to be "empty" (size=0)
            if self.n_input != 0:
                #reshape the data
                u = inputData.reshape(self.n_input, 1)
                outputData = outputData.reshape(self.n_output, 1)

                #update the states
                transmission = self.calculateLinearNetworkTransmissions(u, x)
                x *= (1.0-self._leakingRate)
                x += self._leakingRate*self._activation(transmission +
                     B.dot(self._W_feedback, B.vstack((self.output_bias, outputData))) + (B.rand()-0.5)*self.noise_level)

                return u
            else:
                #reshape the data
                outputData = outputData.reshape(self.n_output, 1)
                #update the states
                transmission = B.dot(self._W, x)
                x *= (1.0-self._leakingRate)
                x += self._leakingRate*self._activation(transmission + B.dot(self._W_feedback, B.vstack((self.output_bias, outputData))) +
                     (B.rand()-0.5)*self.noise_level)

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
