"""
    Implementation of the general ESN model.
"""

from .BaseESN import BaseESN

from easyesn import backend as B

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import progressbar

from .optimizers import GradientOptimizer
from .optimizers import GridSearchOptimizer


class PredictionESN(BaseESN):
    def __init__(
        self,
        n_input,
        n_reservoir,
        n_output,
        spectralRadius=1.0,
        noiseLevel=0.0,
        inputScaling=None,
        leakingRate=1.0,
        feedbackScaling=1.0,
        reservoirDensity=0.2,
        randomSeed=None,
        out_activation=lambda x: x,
        out_inverse_activation=lambda x: x,
        weightGeneration="naive",
        bias=1.0,
        outputBias=1.0,
        feedback=False,
        outputInputScaling=1.0,
        inputDensity=1.0,
        solver="pinv",
        regressionParameters={},
        activation=B.tanh,
        activationDerivative=lambda x: 1.0 / B.cosh(x) ** 2,
    ):
        """ ESN that predicts (steps of) a time series based on a time series.

            Args:
                n_input : Dimensionality of the input.
                n_reservoir : Number of units in the reservoir.
                n_output : Dimensionality of the output.
                spectralRadius : Spectral radius of the reservoir's connection/weight matrix.
                noiseLevel : Magnitude of noise that is added to the input while fitting to prevent overfitting.
                inputScaling : Scaling factor of the input.
                leakingRate : Convex combination factor between 0 and 1 that weights current and new state value.
                feedbackScaling : Rescaling factor of the output-to-input feedback in the update process.
                reservoirDensity : Percentage of non-zero weight connections in the reservoir.
                randomSeed : Seed for random processes, e.g. weight initialization.
                out_activation : Final activation function (i.e. activation function of the output).
                out_inverse_activation : Inverse of the final activation function
                weightGeneration : Algorithm to generate weight matrices. Choices: naive, SORM, advanced, custom
                bias : Size of the bias added for the internal update process.
                outputBias : Size of the bias added for the final linear regression of the output.
                feedback : Include output-input feedback in the ESN.
                outputInputScaling : Rescaling factor for the input of the ESN for the regression.
                inputDensity : Percentage of non-zero weights in the input-to-reservoir weight matrix.
                solver : Algorithm to find output matrix. Choices: pinv, lsqr.
                regressionParameters : Arguments to the solving algorithm. For LSQR this controls the L2 regularization.
                activation : (Non-linear) Activation function.
                activationDerivative : Derivative of the activation function.
        """

        super(PredictionESN, self).__init__(
            n_input=n_input,
            n_reservoir=n_reservoir,
            n_output=n_output,
            spectralRadius=spectralRadius,
            noiseLevel=noiseLevel,
            inputScaling=inputScaling,
            leakingRate=leakingRate,
            feedbackScaling=feedbackScaling,
            reservoirDensity=reservoirDensity,
            randomSeed=randomSeed,
            feedback=feedback,
            out_activation=out_activation,
            out_inverse_activation=out_inverse_activation,
            weightGeneration=weightGeneration,
            bias=bias,
            outputBias=outputBias,
            outputInputScaling=outputInputScaling,
            inputDensity=inputDensity,
            activation=activation,
            activationDerivative=activationDerivative,
        )

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

    def fit(
        self,
        inputData,
        outputData,
        transientTime="AutoReduce",
        transientTimeCalculationEpsilon=1e-3,
        transientTimeCalculationLength=20,
        verbose=0,
    ):
        # check the input data
        if self.n_input != 0:
            if len(inputData.shape) == 3 and len(outputData.shape) > 1:
                # multiple time series are used with a shape (timeseries, time, dimension) -> (timeseries, time, dimension)
                if inputData.shape[0] != outputData.shape[0]:
                    raise ValueError(
                        "Amount of input and output datasets is not equal - {0} != {1}".format(
                            inputData.shape[0], outputData.shape[0]
                        )
                    )
                if inputData.shape[1] != outputData.shape[1]:
                    raise ValueError(
                        "Amount of input and output time steps is not equal - {0} != {1}".format(
                            inputData.shape[1], outputData.shape[1]
                        )
                    )
            else:
                if inputData.shape[0] != outputData.shape[0]:
                    raise ValueError(
                        "Amount of input and output time steps is not equal - {0} != {1}".format(
                            inputData.shape[0], outputData.shape[0]
                        )
                    )
        else:
            if inputData is not None:
                raise ValueError(
                    "n_input has been set to zero. Therefore, the given inputData will not be used."
                )

        if inputData is not None:
            inputData = B.array(inputData)
        if outputData is not None:
            outputData = B.array(outputData)

        # reshape the input/output data to have the shape (timeseries, time, dimension)
        if len(outputData.shape) <= 2:
            outputData = outputData.reshape((1, -1, self.n_output))
        if inputData is not None:
            if len(inputData.shape) <= 2:
                inputData = inputData.reshape((1, -1, self.n_input))

        self.resetState()

        # Automatic transient time calculations
        if transientTime == "Auto":
            transientTime = self.calculateTransientTime(
                inputData[0],
                outputData[0],
                transientTimeCalculationEpsilon,
                transientTimeCalculationLength,
            )
        if transientTime == "AutoReduce":
            if (inputData is None and outputData.shape[2] == 1) or inputData.shape[
                2
            ] == 1:
                transientTime = self.calculateTransientTime(
                    inputData[0],
                    outputData[0],
                    transientTimeCalculationEpsilon,
                    transientTimeCalculationLength,
                )
                transientTime = self.reduceTransientTime(
                    inputData[0], outputData[0], transientTime
                )
            else:
                print(
                    "Transient time reduction is supported only for 1 dimensional input."
                )

        if inputData is not None:
            partialLength = inputData.shape[1] - transientTime
            totalLength = inputData.shape[0] * partialLength
            timeseriesCount = inputData.shape[0]
        elif outputData is not None:
            partialLength = outputData.shape[1] - transientTime
            totalLength = outputData.shape[0] * partialLength
            timeseriesCount = outputData.shape[0]
        else:
            raise ValueError("Either input or output data must not to be None")

        self._X = B.empty((1 + self.n_input + self.n_reservoir, totalLength))

        if verbose > 0:
            bar = progressbar.ProgressBar(
                max_value=totalLength, redirect_stdout=True, poll_interval=0.0001
            )
            bar.update(0)

        for i in range(timeseriesCount):
            if inputData is not None:
                self._X[
                    :, i * partialLength : (i + 1) * partialLength
                ] = self.propagate(
                    inputData[i], outputData[i], transientTime, verbose - 1
                )
            else:
                self._X[
                    :, i * partialLength : (i + 1) * partialLength
                ] = self.propagate(None, outputData[i], transientTime, verbose - 1)
            if verbose > 0:
                bar.update(i)
        if verbose > 0:
            bar.finish()

        # define the target values
        Y_target = B.empty((outputData.shape[2], totalLength))
        for i in range(timeseriesCount):
            Y_target[
                :, i * partialLength : (i + 1) * partialLength
            ] = self.out_inverse_activation(outputData[i]).T[:, transientTime:]

        if self._solver == "pinv":
            self._WOut = B.dot(Y_target, B.pinv(self._X))

            # calculate the training prediction now
            train_prediction = self.out_activation((B.dot(self._WOut, self._X)).T)

        elif self._solver == "lsqr":
            X_T = self._X.T
            self._WOut = B.dot(
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

                self._WOut = np.dot(B, A)
                self._WOut = self._WOut.T
            """

            # calculate the training prediction now
            train_prediction = self.out_activation(B.dot(self._WOut, self._X).T)

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

            self._ridgeSolver.fit(self._X.T, Y_target.T.flatten())

            # calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(self._X.T))

        # calculate the training error now
        # flatten the outputData
        outputData = outputData[:, transientTime:, :].reshape(totalLength, -1)
        training_error = B.sqrt(B.mean((train_prediction - outputData) ** 2))
        return training_error

    """
        Use the ESN in the generative mode to generate a signal autonomously.
    """

    def generate(
        self,
        n,
        inputData=None,
        initialOutputData=None,
        continuation=True,
        initialData=None,
        update_processor=lambda x: x,
        verbose=0,
    ):
        # initialOutputData is the output of the last step BEFORE the generation shall start, e.g. the last step of the training's output

        # check the input data
        # if (self.n_input != self.n_output):
        #    raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        if inputData is not None:
            inputData = B.array(inputData)

        if initialOutputData is not None:
            initialOutputData = B.array(initialOutputData)

        if initialData is not None:
            initialData = B.array(initialData)

        if initialOutputData is None and initialData is None:
            raise ValueError(
                "Either intitialOutputData or initialData must be different from None, as the network needs an initial output value"
            )

        if initialOutputData is None and initialData is not None:
            initialOutputData = initialData[1][-1]

        if inputData is not None:
            inputData = B.array(inputData)
        if initialData is not None:
            initialData = B.array(initialData)

        # let some input run through the ESN to initialize its states from a new starting value
        if not continuation:
            self._x = B.zeros(self._x.shape)

            if initialData is not None:
                if type(initialData) is tuple:
                    initialDataInput, initialDataOutput = initialData
                    if initialDataInput is not None and len(initialDataInput) != len(
                        initialDataOutput
                    ):
                        raise ValueError(
                            "Length of the inputData and the outputData of the initialData tuple do not match."
                        )
                else:
                    raise ValueError(
                        "initialData has to be a tuple consisting out of the input and the output data."
                    )

                for t in range(initialDataInput.shape[0]):
                    super(PredictionESN, self).update(
                        initialDataInput[t], initialDataOutput[t]
                    )

        if self.n_input != 0:
            if inputData is None:
                raise ValueError("inputData must not be None.")
            elif len(inputData) < n:
                raise ValueError("Length of inputData has to be >= n.")

        _, Y = self.propagate(
            inputData,
            None,
            verbose=verbose,
            steps=n,
            previousOutputData=initialOutputData,
        )
        Y = update_processor(Y)

        # return the result
        return Y.T

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """

    def predict(
        self,
        inputData,
        continuation=True,
        initialData=None,
        update_processor=lambda x: x,
        verbose=0,
    ):
        inputData = B.array(inputData)

        # let some input run through the ESN to initialize its states from a new starting value
        if not continuation:
            self._x = B.zeros(self._x.shape)

            if initialData is not None:
                if self._WFeedback is None:
                    for t in range(initialData.shape[0]):
                        super(PredictionESN, self).update(initialData[t])
                else:
                    if type(initialData) is tuple:
                        initialDataInput, initialDataOutput = initialData
                        if len(initialDataInput) != len(initialDataOutput):
                            raise ValueError(
                                "Length of the inputData and the outputData of the initialData tuple do not match."
                            )
                    else:
                        raise ValueError(
                            "initialData has to be a tuple consisting out of the input and the output data."
                        )

                    super(PredictionESN, self).update(
                        initialDataInput[t], initialDataOutput[t]
                    )

        X = self.propagate(inputData, verbose=verbose)

        if self._WFeedback is not None:
            X, _ = X

        # calculate the prediction using the trained model
        if self._solver in [
            "sklearn_auto",
            "sklearn_lsqr",
            "sklearn_sag",
            "sklearn_svd",
            "sklearn_svr",
        ]:
            Y = self._ridgeSolver.predict(X.T).reshape((self.n_output, -1))
        else:
            Y = B.dot(self._WOut, X)

        # apply the output activation function
        Y = update_processor(self.out_activation(Y))

        # return the result
        return Y.T

    def optimize(
        self, trainingInput, trainingOutput, validationInput, validationOutput, verbose
    ):
        gridSearch = GridSearchOptimizer()
        gradientOptimizer = GradientOptimizer()
        pipe = Pipeline(gridSearch, gradientOptimizer)

        pipe.fit(
            trainingInput, trainingOutput, validationInput, validationOutput, verbose
        )
