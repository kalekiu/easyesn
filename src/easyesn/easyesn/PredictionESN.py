"""
    Implementation of the general ESN model.
"""

import numpy as np
import numpy.random as rnd
from .BaseESN import BaseESN

from . import backend as B

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import progressbar

from .optimizers import GradientOptimizer
from .optimizers import GridSearchOptimizer


class PredictionESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectralRadius=1.0, noiseLevel=0.0, inputScaling=None,
                 leakingRate=1.0, feedbackScaling = 1.0, reservoirDensity=0.2, randomSeed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weightGeneration='naive', bias=1.0, outputBias=1.0, feedback = False,
                 outputInputScaling=1.0, input_density=1.0, solver='pinv', regressionParameters={}, activation = B.tanh, activationDerivation=lambda x: 1.0/B.cosh(x)**2):

        super(PredictionESN, self).__init__(n_input=n_input, n_reservoir=n_reservoir, n_output=n_output, spectralRadius=spectralRadius,
                                  noiseLevel=noiseLevel, inputScaling=inputScaling, leakingRate=leakingRate, feedbackScaling = feedbackScaling, reservoirDensity=reservoirDensity,
                                  randomSeed=randomSeed, feedback = feedback, out_activation=out_activation, out_inverse_activation=out_inverse_activation,
                                  weightGeneration=weightGeneration, bias=bias, outputBias=outputBias, outputInputScaling=outputInputScaling,
                                  input_density=input_density, activation=activation, activationDerivation=activationDerivation)


        self._solver = solver
        self._regressionParameters = regressionParameters

        self._x = B.zeros((self.n_reservoir, 1))

        """
            allowed values for the solver:
                pinv
                lsqr (will only be used in the thesis)

                sklearn_auto
                sklearn_svd
                sklearn_cholesky
                sklearn_lsqr
                sklearn_sag
        """


    """
        Fits the ESN so that by applying the inputData the outputData will be produced.
    """
    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        #check the input data
        if self.n_input != 0:
            if inputData.shape[0] != outputData.shape[0]:
                raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))
            trainLength = inputData.shape[0]
        else:
            if inputData is not None:
                raise ValueError("n_input has been set to zero. Therefore, the given inputData will not be used.")
            trainLength = outputData.shape[0]

        
        self.resetState()

        self._X = self.propagate(inputData, outputData, transientTime, verbose)


        #define the target values
        Y_target = self.out_inverse_activation(outputData).T[:,transientTime:]

        if (self._solver == "pinv"):
            self._WOut = B.dot(Y_target, B.pinv(self._X))

            #calculate the training prediction now
            train_prediction = self.out_activation((B.dot(self._WOut, self._X)).T)

        elif (self._solver == "lsqr"):
            X_T = self._X.T
            self._WOut = B.dot(B.dot(Y_target, X_T),B.inv(B.dot(self._X,X_T) + self._regressionParameters[0]*B.identity(1+self.n_input+self.n_reservoir)))

            """
                #alternative represantation of the equation

                Xt = X.T

                A = np.dot(X, Y_target.T)

                B = np.linalg.inv(np.dot(X, Xt)  + regression_parameter*np.identity(1+self.n_input+self.n_reservoir))

                self._WOut = np.dot(B, A)
                self._WOut = self._WOut.T
            """

            #calculate the training prediction now
            train_prediction = self.out_activation(B.dot(self._WOut, self._X).T)

        elif (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
            mode = self._solver[8:]
            params = self._regressionParameters
            params["solver"] = mode
            self._ridgeSolver = Ridge(**params)

            self._ridgeSolver.fit(self._X.T, Y_target.T)

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        elif (self._solver in ["sklearn_svr", "sklearn_svc"]):
            self._ridgeSolver = SVR(**self._regressionParameters)

            self._ridgeSolver.fit(self._X.T, Y_target.T.ravel())

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        #calculate the training error now
        training_error = B.sqrt(B.mean((train_prediction - outputData[transientTime:])**2))
        return training_error


    """
        Use the ESN in the generative mode to generate a signal autonomously.
    """
    def generate(self, n, inputData=None, continuation=True, initialData=None, update_processor=lambda x:x, verbose=0):
        #check the input data
        #if (self.n_input != self.n_output):
        #    raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        #let some input run through the ESN to initialize its states from a new starting value
        if not continuation:
            self._x = B.zeros(self._x.shape)

            if initialData is not None:
                if type(initialData) is tuple:
                    initialDataInput, initialDataOutput = initialData 
                    if initialDataInput is not None and len(initialDataInput) != len(initialDataOutput):
                        raise ValueError("Length of the inputData and the outputData of the initialData tuple do not match.")
                else:
                    raise ValueError("initialData has to be a tuple consisting out of the input and the output data.")

                for t in range(initialDataInput.shape[0]):
                    super(PredictionESN, self).update(initialDataInput[t], initialDataOutput[t])

        predictionLength = n
        if self.n_input != 0:
            if inputData is None:
                raise ValueError("inputData must not be None.")
            elif len(inputData) < n:
                raise ValueError("Length of inputData has to be >= n.")

        _, Y = self.propagate(inputData, None, verbose=verbose, steps=n)
        Y = update_processor(Y)
        
        #return the result
        return Y.T

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, continuation=True, initialData=None, update_processor=lambda x:x, verbose=0):
        #let some input run through the ESN to initialize its states from a new starting value
        if (not continuation):
            self._x = B.zeros(self._x.shape)

            if initialData is not None:
                if self._WFeedback is None:
                    for t in range(initialData.shape[0]):
                        super(PredictionESN, self).update(initialData[t])
                else:   
                    if type(initialData) is tuple:
                        initialDataInput, initialDataOutput = initialData 
                        if (len(initialDataInput) != len(initialDataOutput)):
                            raise ValueError("Length of the inputData and the outputData of the initialData tuple do not match.")
                    else:
                        raise ValueError("initialData has to be a tuple consisting out of the input and the output data.")

                    super(PredictionESN, self).update(initialDataInput[t], initialDataOutput[t])

        predLength = inputData.shape[0]

        X = self.propagate(inputData, verbose=verbose)

        if self._WFeedback is not None:
            X, _ = X

        #calculate the prediction using the trained model
        if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd", "sklearn_svr"]):
            Y = self._ridgeSolver.predict(X.T).reshape((self.n_output, -1))
        else:
            Y = B.dot(self._WOut, X)

        #apply the output activation function
        Y = update_processor(self.out_activation(Y))
     
        #return the result
        return Y.T

    def optimize(self, trainingInput, trainingOutput, validationInput, validationOutput, verbose):
        gridSearch = GridSearch()
        gradientOptimizer = GradientOptimizer()
        pipe = Pipeline(gridSearch, gradientOptimizer)

        pipe.fit(trainingInput, trainingOutput, validationInput, validationOutput, verbose)