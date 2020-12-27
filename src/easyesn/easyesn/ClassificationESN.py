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
from .OneHotEncoder import OneHotEncoder


class ClassificationESN(BaseESN):
    def __init__(
        self,
        n_input,
        n_reservoir,
        n_classes,
        spectralRadius=1.0,
        noiseLevel=0.0,
        inputScaling=None,
        leakingRate=1.0,
        reservoirDensity=0.2,
        randomSeed=None,
        out_activation=lambda x: 0.1 + 0.98 * x / (1 + B.exp(-x)),
        out_inverse_activation=lambda x: B.log((x * 0.98 + 0.01) / (0.99 - x * 0.98)),
        weightGeneration="naive",
        bias=1.0,
        outputBias=1.0,
        outputInputScaling=1.0,
        inputDensity=1.0,
        solver="pinv",
        regressionParameters={},
        activation=B.tanh,
        activationDerivation=lambda x: 1.0 / B.cosh(x) ** 2,
    ):

        super(ClassificationESN, self).__init__(
            n_input=n_input,
            n_reservoir=n_reservoir,
            n_output=n_classes,
            spectralRadius=spectralRadius,
            noiseLevel=noiseLevel,
            inputScaling=inputScaling,
            leakingRate=leakingRate,
            reservoirDensity=reservoirDensity,
            randomSeed=randomSeed,
            out_activation=out_activation,
            out_inverse_activation=out_inverse_activation,
            weightGeneration=weightGeneration,
            bias=bias,
            outputBias=outputBias,
            outputInputScaling=outputInputScaling,
            inputDensity=inputDensity,
            activation=activation,
            activationDerivation=activationDerivation,
        )

        self._solver = solver
        self._regressionParameters = regressionParameters
        self._oneHotEncoder = OneHotEncoder()

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

    def fit(
        self,
        inputData,
        outputData,
        transientTime=0,
        transientTimeCalculationEpsilon=1e-3,
        transientTimeCalculationLength=20,
        verbose=0,
    ):
        # check the input data
        if inputData.shape[0] != outputData.shape[0]:
            raise ValueError(
                "Amount of input and output datasets is not equal - {0} != {1}".format(
                    inputData.shape[0], outputData.shape[0]
                )
            )

        nSequences = inputData.shape[0]
        trainingLength = inputData.shape[1]

        if outputData.shape[1] > 1 and outputData.shape[1] != self.n_output:
            raise ValueError(
                "The outputData has the shape {0}, which indicates that it is already one hot encoded, "
                "but it does not match the number of classes ({1}) specified in the constructur.".format(
                    outputData.shape, self.n_output
                )
            )

        if outputData.shape[1] == 1 and len(np.unique(outputData)) > 2:
            outputData = self._oneHotEncoder.transform(outputData)

        self._x = B.zeros((self.n_reservoir, 1))

        # Automatic transient time calculations
        if transientTime == "Auto":
            transientTime = self.calculateTransientTime(
                inputData,
                outputData,
                transientTimeCalculationEpsilon,
                transientTimeCalculationLength,
            )
        if transientTime == "AutoReduce":
            if (inputData is None and outputData.shape[1] == 1) or inputData.shape[
                1
            ] == 1:
                transientTime = self.calculateTransientTime(
                    inputData,
                    outputData,
                    transientTimeCalculationEpsilon,
                    transientTimeCalculationLength,
                )
                transientTime = self.reduceTransientTime(
                    inputData, outputData, transientTime
                )
            else:
                print(
                    "Transient time reduction is supported only for 1 dimensional input."
                )

        self._X = B.zeros(
            (
                1 + self.n_input + self.n_reservoir,
                nSequences * (trainingLength - transientTime),
            )
        )
        Y_target = B.zeros(
            (self.n_output, (trainingLength - transientTime) * nSequences)
        )

        if verbose > 0:
            bar = progressbar.ProgressBar(
                max_value=len(inputData), redirect_stdout=True, poll_interval=0.0001
            )
            bar.update(0)

        for n in range(len(inputData)):
            self._x = B.zeros((self.n_reservoir, 1))
            self._X[
                :,
                n
                * (trainingLength - transientTime) : (n + 1)
                * (trainingLength - transientTime),
            ] = self.propagate(inputData[n], transientTime, verbose=0)
            # set the target values
            Y_target[
                :,
                n
                * (trainingLength - transientTime) : (n + 1)
                * (trainingLength - transientTime),
            ] = (
                np.tile(
                    self.out_inverse_activation(outputData[n]),
                    trainingLength - transientTime,
                )
                .reshape(-1, self.n_output)
                .T
            )

            if verbose > 0:
                bar.update(n)

        if verbose > 0:
            bar.finish()

        if self._solver == "pinv":
            self._W_out = B.dot(Y_target, B.pinv(self._X))

            # calculate the training prediction now
            train_prediction = self.out_activation((B.dot(self._W_out, self._X)).T)

        elif self._solver == "lsqr":
            X_T = self._X.T
            self._W_out = B.dot(
                B.dot(Y_target, X_T),
                B.inv(
                    B.dot(self._X, X_T)
                    + self._regressionParameters[0]
                    * B.identity(1 + self.n_input + self.n_reservoir)
                ),
            )

            """
                #alternative represantation of the equation

                Xt = X.T

                A = np.dot(X, Y_target.T)

                B = np.linalg.inv(np.dot(X, Xt)  + regression_parameter*np.identity(1+self.n_input+self.n_reservoir))

                self._W_out = np.dot(B, A)
                self._W_out = self._W_out.T
            """

            # calculate the training prediction now
            train_prediction = self.out_activation(B.dot(self._W_out, self._X).T)

        elif self._solver in [
            "sklearn_auto",
            "sklearn_lsqr",
            "sklearn_sag",
            "sklearn_svd",
        ]:
            mode = self._solver[8:]
            params = self._regressionParameters
            params["solver"] = mode
            self._ridgeSolver = Ridge(**params)

            self._ridgeSolver.fit(self._X.T, Y_target.T)

            # calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        elif self._solver in ["sklearn_svr", "sklearn_svc"]:
            self._ridgeSolver = SVR(**self._regressionParameters)

            self._ridgeSolver.fit(self._X.T, Y_target.T.ravel())

            # calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        train_prediction = train_prediction[::trainingLength]

        # calculate the training error now
        training_error = B.sqrt(B.mean((train_prediction - outputData) ** 2))
        return training_error

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """

    def predict(
        self, inputData, update_processor=lambda x: x, transientTime=0, verbose=0
    ):
        if len(inputData.shape) == 1:
            inputData = inputData[None, :]

        predictionLength = inputData.shape[1]

        Y = B.empty((inputData.shape[0], self.n_output))

        if verbose > 0:
            bar = progressbar.ProgressBar(
                max_value=inputData.shape[0], redirect_stdout=True, poll_interval=0.0001
            )
            bar.update(0)

        for n in range(inputData.shape[0]):
            # reset the state
            self._x = B.zeros(self._x.shape)

            X = self.propagate(inputData[n], transientTime)
            # calculate the prediction using the trained model
            if self._solver in [
                "sklearn_auto",
                "sklearn_lsqr",
                "sklearn_sag",
                "sklearn_svd",
                "sklearn_svr",
            ]:
                y = self._ridgeSolver.predict(X.T).reshape((self.n_output, -1))
            else:
                y = B.dot(self._W_out, X)

            Y[n] = np.mean(y, 1)

            if verbose > 0:
                bar.update(n)

        if verbose > 0:
            bar.finish()

        # return the result
        result = B.zeros(Y.shape)
        for i, n in enumerate(B.argmax(Y, 1)):
            result[i, n] = 1.0
        return result
