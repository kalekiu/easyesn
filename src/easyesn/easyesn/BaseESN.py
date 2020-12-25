"""
    Basic implementation of an ESN.
"""

#from __future__ import absolute_import

import numpy as np
import dill as pickle
import progressbar
from . import helper as hp

#import backend as B

from . import backend as B

class BaseESN(object):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectralRadius=1.0, noiseLevel=0.01, inputScaling=None,
                 leakingRate=1.0, feedbackScaling = 1.0, reservoirDensity=0.2, randomSeed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weightGeneration='naive', bias=1.0, outputBias=1.0, outputInputScaling=1.0,
                 feedback=False, inputDensity=1.0, activation = B.tanh, activationDerivation=lambda x: 1.0/B.cosh(x)**2):

        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output

        self._spectralRadius = spectralRadius
        self._noiseLevel = noiseLevel
        self._reservoirDensity = reservoirDensity
        self._leakingRate = leakingRate
        self._feedbackScaling = feedbackScaling
        self.inputDensity = inputDensity
        self._activation = activation
        self._activationDerivation = activationDerivation
        self._inputScaling = inputScaling

        if self._inputScaling is None:
            self._inputScaling = 1.0
        if np.isscalar(self._inputScaling):
            self._inputScaling = B.ones(n_input) * self._inputScaling
        else:
            if len(self._inputScaling) != self.n_input:
                raise ValueError("Dimension of inputScaling ({0}) does not match the input data dimension ({1})".format(len(self._inputScaling), n_input))
            self._inputScaling = inputScaling

        self._expandedInputScaling = B.vstack((B.array(1.0), self._inputScaling.reshape(-1,1))).flatten()

        self.out_activation = out_activation
        self.out_inverse_activation = out_inverse_activation

        if randomSeed is not None:
            B.seed(randomSeed)
            np.random.seed(randomSeed)

        self._bias = bias
        self._outputBias = outputBias
        self._outputInputScaling = outputInputScaling
        self._createReservoir(weightGeneration, feedback)


    def setSpectralRadius(self, newSpectralRadius):
        self._W = self._W * ( newSpectralRadius / self._spectralRadius )
        self._spectralRadius = newSpectralRadius
        #TODO numerical instability

    def setLeakingRate(self, newLeakingRate):
        self._leakingRate = newLeakingRate

    def setInputScaling(self, newInputScaling):
        inputScaling = B.ones(self.n_input) * self._inputScaling
        self._expandedInputScaling = B.vstack((B.array(1.0), inputScaling.reshape(-1, 1))).flatten()
        self._WInput = self._WInput * ( self._expandedInputScaling / self._inputScaling )
        self._inputScaling = newInputScaling

    def setFeedbackScaling(self, newFeedbackScaling):
        self._WFeedback = self._WFeedback * ( newFeedbackScaling / self._feedbackScaling)
        self._feedbackScaling = newFeedbackScaling


    def resetState(self):
        self._x = B.zeros_like(self._x)

    def propagate(self, inputData, outputData=None, transientTime=0, verbose=0, x=None, steps="auto", previousOutputData=None):
        if x is None:
            x = self._x

        inputLength = steps
        if inputData is None:
            if outputData is not None:
                inputLength = len(outputData)
        else:
            inputLength = len(inputData)
        if inputLength == "auto":
            raise ValueError("inputData and outputData are both None. Therefore, steps must not be `auto`.")

        # define states' matrix
        X = B.zeros((1 + self.n_input + self.n_reservoir, inputLength - transientTime))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=inputLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        if self._WFeedback is None:
            #do not distinguish between whether inputData is None or not, as the feedback has been disabled
            #therefore, the input has to be anything but None

            for t in range(inputLength):
                u = self.update(inputData[t], x=x)
                if (t >= transientTime):
                    #add valueset to the states' matrix
                    X[:,t-transientTime] = B.vstack((B.array(self._outputBias), self._outputInputScaling*u, x))[:,0]
                if (verbose > 0):
                    bar.update(t)
        else:
            if outputData is None:
                Y = B.empty((inputLength-transientTime, self.n_output))

            if previousOutputData is None:
                previousOutputData = B.zeros((1, self.n_output))

            if inputData is None:
                for t in range(inputLength):
                    self.update(None, previousOutputData, x=x)
                    if (t >= transientTime):
                        #add valueset to the states' matrix
                        X[:,t-transientTime] = B.vstack((B.array(self._outputBias), x))[:,0]
                    if outputData is None:
                        #calculate the prediction using the trained model
                        if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
                            previousOutputData = self._ridgeSolver.predict(B.vstack((B.array(self._outputBias), self._x)).T)
                        else:
                            previousOutputData = B.dot(self._WOut, B.vstack((B.array(self._outputBias), self._x)))
                        if t >= transientTime:
                            Y[t-transientTime, :] = previousOutputData.ravel()
                    else:
                        previousOutputData = outputData[t]

                    if (verbose > 0):
                        bar.update(t)
            else:
                for t in range(inputLength):
                    u = self.update(inputData[t], previousOutputData, x=x)
                    if (t >= transientTime):
                        #add valueset to the states' matrix
                        X[:,t-transientTime] = B.vstack((B.array(self._outputBias), self._outputInputScaling*u, x))[:,0]
                    if outputData is None:
                        #calculate the prediction using the trained model
                        if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
                            previousOutputData = self._ridgeSolver.predict(B.vstack((B.array(self._outputBias), self._outputInputScaling*u, self._x)).T)
                        else:
                            previousOutputData = B.dot(self._WOut, B.vstack((B.array(self._outputBias), self._outputInputScaling*u, self._x)))
                        Y[t, :] = previousOutputData
                    else:
                        previousOutputData = outputData[t]

                    if (verbose > 0):
                        bar.update(t)

        if (verbose > 0):
            bar.finish()

        if self._WFeedback is not None and outputData is None:
            return X, Y
        else:
            return X

    """
        Generates a random rotation matrix, used in the SORM initilization (see http://ftp.math.uni-rostock.de/pub/preprint/2012/pre12_01.pdf)
    """
    def create_random_rotation_matrix(self):
        h = B.randint(low=0, high=self.n_reservoir)
        k = B.randint(low=0, high=self.n_reservoir)

        phi = B.rand(1)*2*np.pi

        Q = B.identity(self.n_reservoir)
        Q[h, h] = B.cos(phi)
        Q[k, k] = B.cos(phi)

        Q[h, k] = -B.sin(phi)
        Q[k, h] = B.sin(phi)

        return Q

    """
        Internal method to create the matrices W_in, W and W_fb of the ESN
    """
    def _createReservoir(self, weightGeneration, feedback=False, verbose=False):
        #naive generation of the matrix W by using random weights
        if weightGeneration == 'naive':
            #random weight matrix from -0.5 to 0.5
            self._W = B.array(B.rand(self.n_reservoir, self.n_reservoir) - 0.5)

            #set sparseness% to zero
            mask = B.rand(self.n_reservoir, self.n_reservoir) > self._reservoirDensity
            self._W[mask] = 0.0

            _W_eigenvalues = B.abs(B.eigenval(self._W)[0])
            self._W *= self._spectralRadius / B.max(_W_eigenvalues)

        #generation using the SORM technique (see http://ftp.math.uni-rostock.de/pub/preprint/2012/pre12_01.pdf)
        elif weightGeneration == "SORM":
            self._W = B.identity(self.n_reservoir)

            number_nonzero_elements = self._reservoirDensity * self.n_reservoir * self.n_reservoir
            i = 0

            while B.count_nonzero(self._W) < number_nonzero_elements:
                i += 1
                Q = self.create_random_rotation_matrix()
                self._W = Q.dot(self._W)

            self._W *= self._spectralRadius

        #generation using the proposed method of Yildiz
        elif weightGeneration == 'advanced':
            #two create W we must follow some steps:
            #at first, create a W = |W|
            #make it sparse
            #then scale its spectral radius to rho(W) = 1 (according to Yildiz with x(n+1) = (1-a)*x(n)+a*f(...))
            #then change randomly the signs of the matrix

            #random weight matrix from 0 to 0.5

            self._W = B.array(B.rand(self.n_reservoir, self.n_reservoir) / 2)

            #set sparseness% to zero
            mask = B.rand(self.n_reservoir, self.n_reservoir) > self._reservoirDensity
            self._W[mask] = 0.0

            _W_eigenvalue = B.max(B.abs(B.eigvals(self._W)))

            self._W *= self._spectralRadius / _W_eigenvalue

            if verbose:
                M = self._leakingRate*self._W + (1 - self._leakingRate)*B.identity(n=self._W.shape[0])
                M_eigenvalue = B.max(B.abs(B.eigenval(M)[0]))
                print("eff. spectral radius: {0}".format(M_eigenvalue))

            #change random signs
            random_signs = B.power(-1, B.randint(1, 3, (self.n_reservoir,)))

            self._W = B.multiply(self._W, random_signs)
        elif weightGeneration == 'custom':
            pass
        else:
            raise ValueError("The weightGeneration property must be one of the following values: naive, advanced, SORM, custom")

        #check of the user is really using one of the internal methods, or wants to create W by his own
        if (weightGeneration != 'custom'):
            self._createInputMatrix()

        #create the optional feedback matrix
        if feedback:
            self._WFeedback = B.rand(self.n_reservoir, 1 + self.n_output) - 0.5
            self._WFeedback *= self._feedbackScaling
        else:
            self._WFeedback = None

    def _createInputMatrix(self):
        #random weight matrix for the input from -0.5 to 0.5
        self._WInput = B.rand(self.n_reservoir, 1 + self.n_input)-0.5

        #scale the inputDensity to prevent saturated reservoir nodes
        if (self.inputDensity != 1.0):
            #make the input matrix as dense as requested
            input_topology = (B.ones_like(self._WInput) == 1.0)
            nb_non_zero_input = int(self.inputDensity * self.n_input)
            for n in range(self.n_reservoir):
                input_topology[n][B.permutation(B.arange(1+self.n_input))[:nb_non_zero_input]] = False

            self._WInput[input_topology] = 0.0

        self._WInput = self._WInput * self._expandedInputScaling

    def calculateLinearNetworkTransmissions(self, u, x=None):
        if x is None:
            x = self._x
        return B.dot(self._WInput, B.vstack((B.array(self._bias), u))) + B.dot(self._W, x)

    """
        Updates the inner states. Returns the UNSCALED but reshaped input of this step.
    """
    def update(self, inputData, outputData=None, x=None):
        if x is None:
            x = self._x

        if self._WFeedback is None:
            #reshape the data
            u = inputData.reshape(self.n_input, 1)

            #update the states
            transmission = self.calculateLinearNetworkTransmissions(u, x)
            x *= (1.0-self._leakingRate)
            x += self._leakingRate * self._activation(transmission + (np.random.rand()-0.5)*self._noiseLevel)

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
                                                        B.dot(self._WFeedback, B.vstack((B.array(self._outputBias), outputData))) + (np.random.rand()-0.5)*self._noiseLevel)

                return u
            else:
                #reshape the data
                outputData = outputData.reshape(self.n_output, 1)
                #update the states
                transmission = B.dot(self._W, x)
                x *= (1.0-self._leakingRate)
                x += self._leakingRate*self._activation(transmission + B.dot(self._WFeedback, B.vstack((B.array(self._outputBias), outputData))) +
                                                        (np.random.rand()-0.5)*self._noiseLevel)

                return B.empty((0, 1))

    def calculateTransientTime(self, inputs, outputs, epsilon, proximityLength = None):
        # inputs: input of reserovoir
        # outputs: output of reservoir
        # epsilon: given constant
        # proximity length: number of steps for which all states have to be epsilon close to declare convergance
        # initializes two initial states as far as possible from each other in [-1,1] regime and tests when they converge-> this is transient time

        length = inputs.shape[0] if inputs is not None else outputs.shape[0]
        if proximityLength is None:
            proximityLength = int(length * 0.1)
            if proximityLength < 3:
                proximityLength = 3

        initial_x = B.empty((2, self.n_reservoir, 1))
        initial_x[0] = - B.ones((self.n_reservoir, 1))
        initial_x[1] = B.ones((self.n_reservoir, 1))

        countedConsecutiveSteps = 0
        length = inputs.shape[0] if inputs is not None else outputs.shape[0]
        for t in range(length):
            if B.max(B.ptp(initial_x, axis=0)) < epsilon:
                if countedConsecutiveSteps >= proximityLength:
                    return t - proximityLength
                else:
                    countedConsecutiveSteps += 1
            else:
                countedConsecutiveSteps = 0

            u = inputs[t].reshape(-1, 1) if inputs is not None else None
            o = outputs[t].reshape(-1, 1) if outputs is not None else None
            for i in range(initial_x.shape[0]):
                self.update(u, o, initial_x[i])

        #transient time could not be determined
        raise ValueError("Transient time could not be determined - maybe the proximityLength is too big.")


    def reduceTransientTime(self, inputs, outputs, initialTransientTime, epsilon = 1e-3, proximityLength = 50):
        # inputs: input of reserovoir
        # outputs: output of reservoir
        # epsilon: given constant
        # proximity length: number of steps for which all states have to be epsilon close to declare convergance
        # initialTransientTime: transient time with calculateTransientTime() method estimated
        # finds initial state with lower transient time and sets internal state to this state
        # returns the new transient time by calculating the convergence time of initial states found with SWD and Equilibrium method

        def getEquilibriumState(inputs, outputs, epsilon = 1e-3):
            # inputs: input of reserovoir
            # outputs: output of reservoir
            # epsilon: given constant
            # returns the equilibrium state when esn is fed with the first state of input
            x = B.empty((2, self.n_reservoir, 1))
            while not B.max(B.ptp(x, axis=0)) < epsilon:
                x[0] = x[1]
                u = inputs[0].reshape(-1, 1) if inputs is not None else None
                o = outputs[0].reshape(-1, 1) if outputs is not None else None
                self.update(u, o, x[1])

            return x[1]

        def getStateAtGivenPoint(inputs, outputs, targetTime):
            # inputs: input of reserovoir
            # outputs: output of reservoir
            # targetTime: time at which the state is wanted
            # propagates the inputs/outputs till given point in time and returns the state of the reservoir at this point
            x = B.zeros((self.n_reservoir, 1))

            length = inputs.shape[0] if inputs is not None else outputs.shape[0]
            length = min(length, targetTime)
            for t in range(length):
                u = inputs[t].reshape(-1, 1) if inputs is not None else None
                o = outputs[t].reshape(-1, 1) if outputs is not None else None
                self.update(u, o, x)

            return x

        length = inputs.shape[0] if inputs is not None else outputs.shape[0]
        if proximityLength is None:
            proximityLength = int(length * 0.1)
            if proximityLength < 3:
                proximityLength = 3

        x = B.empty((2, self.n_reservoir, 1))
        equilibriumState = getEquilibriumState(inputs, outputs)

        if inputs is None:
            swdPoint, _ = hp.SWD(outputs, int(initialTransientTime*0.8))
        else:
            swdPoint, _ = hp.SWD(inputs, int(initialTransientTime * 0.8))

        swdState = getStateAtGivenPoint(inputs, outputs, swdPoint)

        x[0] = equilibriumState
        x[1] = swdState

        transientTime = 0

        countedConsecutiveSteps = 0
        for t in range(length):
            if B.max(B.ptp(x, axis=0)) < epsilon:
                countedConsecutiveSteps += 1
                if countedConsecutiveSteps > proximityLength:
                    transientTime = t - proximityLength
                    break
            else:
                countedConsecutiveSteps = 0

            u = inputs[t].reshape(-1, 1) if inputs is not None else None
            o = outputs[t].reshape(-1, 1) if outputs is not None else None
            for i in range(x.shape[0]):
                self.update(u, o, x[i])

        self._x = x[0]
        return transientTime

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
