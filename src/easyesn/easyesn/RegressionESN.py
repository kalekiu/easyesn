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


class RegressionESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectralRadius=1.0, noise_level=0.0, inputScaling=None,
                 leakingRate=1.0, sparseness=0.2, random_seed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weight_generation='naive', bias=1.0, output_bias=1.0,
                 outputInputScaling=1.0, input_density=1.0, solver='pinv', regression_parameters={}, activation = B.tanh):

        super(RegressionESN, self).__init__(n_input=n_input, n_reservoir=n_reservoir, n_output=n_output, spectralRadius=spectralRadius,
                                  noise_level=noise_level, inputScaling=inputScaling, leakingRate=leakingRate, sparseness=sparseness,
                                  random_seed=random_seed, out_activation=out_activation, out_inverse_activation=out_inverse_activation,
                                  weight_generation=weight_generation, bias=bias, output_bias=output_bias, outputInputScaling=outputInputScaling,
                                  input_density=input_density, activation=activation)


        self._solver = solver
        self._regression_parameters = regression_parameters

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
        Fits the ESN so that by applying a time series out of inputData the outputData will be produced.

    """
    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        #check the input data
        if inputData.shape[0] != outputData.shape[0]:
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        nSequences = inputData.shape[0]
        trainingLength = inputData.shape[1]

        self._x = B.zeros((self.n_reservoir,1))

        self._X = B.zeros((1 + self.n_input + self.n_reservoir, nSequences*(trainLength-transientTime)))
        Y_target = B.zeros((trainingLength-transientTime, nSequences))

        for n in len(inputData):
            self._x = B.zeros((self.n_reservoir, 1))
            self._X[:, n*(trainLength-transientTime):(n+1)*(trainLength-transientTime)] = self.propagate(inputData, trainLength, transientTime, verbose)
            #set the target values
            self.Y_target[:, n] = np.tile(self.out_inverse_activation(outputData), trainLength-transientTime).T

        if (self._solver == "pinv"):
            self._W_out = B.dot(Y_target, B.pinv(self._X))

            #calculate the training prediction now
            train_prediction = self.out_activation((B.dot(self._W_out, self._X)).T)

        elif (self._solver == "lsqr"):
            X_T = self._X.T
            self._W_out = B.dot(B.dot(Y_target, X_T),B.inv(B.dot(self._X,X_T) + self._regression_parameters[0]*B.identity(1+self.n_input+self.n_reservoir)))

            """
                #alternative represantation of the equation

                Xt = X.T

                A = np.dot(X, Y_target.T)

                B = np.linalg.inv(np.dot(X, Xt)  + regression_parameter*np.identity(1+self.n_input+self.n_reservoir))

                self._W_out = np.dot(B, A)
                self._W_out = self._W_out.T
            """

            #calculate the training prediction now
            train_prediction = self.out_activation(B.dot(self._W_out, self._X).T)

        elif (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
            mode = self._solver[8:]
            params = self._regression_parameters
            params["solver"] = mode
            self._ridgeSolver = Ridge(**params)

            self._ridgeSolver.fit(self._X.T, Y_target.T)

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        elif (self._solver in ["sklearn_svr", "sklearn_svc"]):
            self._ridgeSolver = SVR(**self._regression_parameters)

            self._ridgeSolver.fit(self._X.T, Y_target.T.ravel())

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        train_prediction = np.mean(train_prediction, 0)

        #calculate the training error now
        training_error = B.sqrt(B.mean((train_prediction - outputData.T)**2))
        return training_error


    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, update_processor=lambda x:x, transientTime, verbose=0):
        if (len(inputData.shape)) == 1)
            inputData = inputData[None, :]

        predictionLength = inputData.shape[1]

        Y = B.empty((n, self.n_output))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=predLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for n in range(inputData.shape[0]):
            #reset the state
            self._x = B.zeros(self._x.shape)

            X = self.propagate(inputData[n], transientTime)
            #calculate the prediction using the trained model
            if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd", "sklearn_svr"]):
                y = self._ridgeSolver.predict(X.T).reshape((self.n_output, -1))
            else:
                y = B.dot(self._W_out, X)

            Y[n] = np.mean(y, 0)

            if verbose > 0:
                bar.update(n*(predictionLength-transientTime))


        if verbose > 0:
            bar.finish()

        #return the result
        return Y
