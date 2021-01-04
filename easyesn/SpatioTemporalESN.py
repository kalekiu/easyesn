"""
    Implementation of the Spatio-Temporal ESN model.
    See `Observing spatio-temporal dynamics of excitable
    media using reservoir computing` by Zimmermann et al.
    for details.
"""

import numpy as np
from easyesn.BaseESN import BaseESN
from easyesn import backend as B
import progressbar
import sys

# we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
from multiprocess import (
    Manager,
    Pool,
    cpu_count,
)


__all__ = ["SpatioTemporalESN"]


class PredictionArrayIterator:
    def __init__(self, array, jobs, width, stride, stesn):
        self.array = array
        self.jobs = jobs
        self._filterWidth = width
        self._stride = stride
        self.current = 0
        self.stesn = stesn

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.jobs)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            try:
                indices = self.jobs[self.current]
                y, x = indices
                self.current += 1

                inData = self.array[
                    :,
                    y - self._filterWidth : y + self._filterWidth + 1,
                    x - self._filterWidth : x + self._filterWidth + 1,
                ][:, :: self._stride, :: self._stride].reshape(len(self.array), -1)

                state = self.stesn._xs[
                    self.stesn._uniqueIDFromIndices(
                        [x - self._filterWidth for x in indices]
                    )
                ]

                return inData, indices, state
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exc()
                return 0


class FittingArrayIterator:
    def __init__(self, array, output_array, jobs, width, stride, stesn):
        self.array = array
        self.output_array = output_array
        self.jobs = jobs
        self._filterWidth = width
        self._stride = stride
        self.current = 0
        self.stesn = stesn

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.jobs)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            try:
                indices = self.jobs[self.current]
                y, x = indices
                self.current += 1

                inData = self.array[
                    :,
                    :,
                    y - self._filterWidth : y + self._filterWidth + 1,
                    x - self._filterWidth : x + self._filterWidth + 1,
                ][:, :, :: self._stride, :: self._stride].reshape(
                    *self.array.shape[:2], -1
                )

                outData = self.output_array[
                    :, :, y - self._filterWidth, x - self._filterWidth
                ].reshape(len(self.output_array), -1, 1)

                state = self.stesn._xs[
                    self.stesn._uniqueIDFromIndices(
                        [x - self._filterWidth for x in indices]
                    )
                ]

                return inData, outData, indices, state
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exc()
                return 0


class SpatioTemporalESN(BaseESN):
    def __init__(
        self,
        inputShape,
        n_reservoir,
        filterSize=1,
        stride=1,
        borderMode="mirror",
        nWorkers="auto",
        spectralRadius=1.0,
        noiseLevel=0.0,
        inputScaling=None,
        leakingRate=1.0,
        reservoirDensity=0.2,
        randomSeed=None,
        averageOutputWeights=True,
        out_activation=lambda x: x,
        out_inverse_activation=lambda x: x,
        weightGeneration="naive",
        bias=1.0,
        outputBias=1.0,
        outputInputScaling=1.0,
        inputDensity=1.0,
        solver="pinv",
        regressionParameters={},
        activation=B.tanh,
        activationDerivative=lambda x: 1.0 / B.cosh(x) ** 2,
        chunkSize=16,
    ):
        """ ESN that predicts (steps of) a spatio-temporal time series based on a time series.

            Args:
                inputShape : Shape of the input w/o the time axis, e.g. (W, H) for a 2D input.
                n_reservoir : Number of units in the reservoir.
                filterSize : Size of patches used to predict a single output element.
                stride : Stride between different patches.
                borderMode : How to handle border values. Choices: mirror, padding, edge, wrap.
                nWorkers : Number of CPU threads executed in parallel to solve the problem.
                spectralRadius : Spectral radius of the reservoir's connection/weight matrix.
                noiseLevel : Magnitude of noise that is added to the input while fitting to prevent overfitting.
                inputScaling : Scaling factor of the input.
                leakingRate : Convex combination factor between 0 and 1 that weights current and new state value.
                reservoirDensity : Percentage of non-zero weight connections in the reservoir.
                randomSeed : Seed for random processes, e.g. weight initialization.
                averageOutputWeights : Average output matrices after fitting across all pixels or use a distinct matrix
                                        per pixel. The former assumes homogeneity of the problem across all pixels.
                out_activation : Final activation function (i.e. activation function of the output).
                out_inverse_activation : Inverse of the final activation function
                weightGeneration : Algorithm to generate weight matrices. Choices: naive, SORM, advanced, custom
                bias : Size of the bias added for the internal update process.
                outputBias : Size of the bias added for the final linear regression of the output.
                outputInputScaling : Rescaling factor for the input of the ESN for the regression.
                inputDensity : Percentage of non-zero weights in the input-to-reservoir weight matrix.
                solver : Algorithm to find output matrix. Choices: pinv, lsqr.
                regressionParameters : Arguments to the solving algorithm. For LSQR this controls the L2 regularization.
                activation : (Non-linear) Activation function.
                activationDerivative : Derivative of the activation function.
                chunkSize : Internal parameter for the multi-threading. For long time series this should be reduced to
                            avoid OOM errors/getting stuck and to reduce memory consumption.
        """

        self._averageOutputWeights = averageOutputWeights
        if averageOutputWeights and solver != "lsqr":
            raise ValueError(
                "`averageOutputWeights` can only be set to `True` when `solver` is set to `lsqr` (Ridge Regression)"
            )

        self._borderMode = borderMode
        if not borderMode in ["mirror", "padding", "edge", "wrap"]:
            raise ValueError(
                "`borderMode` must be set to one of the following values: `mirror`, `padding`, `edge` or `wrap`."
            )

        self._regressionParameters = regressionParameters
        self._solver = solver

        n_inputDimensions = len(inputShape)

        if filterSize % 2 == 0:
            raise ValueError("filterSize has to be an odd number (1, 3, 5, ...).")
        self._filterSize = filterSize
        self._filterWidth = int(np.floor(filterSize / 2))
        self._stride = stride

        self._n_input = int(np.power(np.ceil(filterSize / stride), n_inputDimensions))

        self.n_inputDimensions = n_inputDimensions
        self.inputShape = inputShape

        if not self._averageOutputWeights:
            self._WOuts = B.empty(
                (np.prod(inputShape), 1, self._n_input + n_reservoir + 1)
            )
            self._WOut = None
        else:
            self._WOuts = None
            self._WOut = B.zeros((1, self._n_input + n_reservoir + 1))
        self._xs = B.empty((np.prod(inputShape), n_reservoir, 1))

        if nWorkers == "auto":
            self._nWorkers = np.max((cpu_count() - 1, 1))
        else:
            self._nWorkers = nWorkers

        manager = Manager()
        self.sharedNamespace = manager.Namespace()
        if hasattr(self, "fitWorkerID") == False or self.parallelWorkerIDs is None:
            self.parallelWorkerIDs = manager.Queue()
            for i in range(self._nWorkers):
                self.parallelWorkerIDs.put((i))

        self._chunkSize = chunkSize

        super(SpatioTemporalESN, self).__init__(
            n_input=self._n_input,
            n_reservoir=n_reservoir,
            n_output=1,
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
            activationDerivative=activationDerivative,
        )

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

    @staticmethod
    def _isWindows():
        return hasattr(sys, "getwindowsversion")

    def resetState(self, index=None):
        if index is None:
            self._x = B.zeros((self._nWorkers, self.n_reservoir, 1))
        else:
            self._x[index] = B.zeros((self.n_reservoir, 1))

    def _embedInputData(self, inputData):
        rank = len(inputData.shape) - 2

        if self._borderMode == "mirror":
            modifiedInputData = np.pad(
                inputData,
                tuple(
                    [(0, 0), (0, 0)] + [(self._filterWidth, self._filterWidth)] * rank
                ),
                mode="symmetric",
            )
        elif self._borderMode == "padding":
            modifiedInputData = np.pad(
                inputData,
                tuple(
                    [(0, 0), (0, 0)] + [(self._filterWidth, self._filterWidth)] * rank
                ),
                mode="constant",
                constant_values=0,
            )
        elif self._borderMode == "edge":
            modifiedInputData = np.pad(
                inputData,
                tuple(
                    [(0, 0), (0, 0)] + [(self._filterWidth, self._filterWidth)] * rank
                ),
                mode="edge",
            )
        elif self._borderMode == "wrap":
            modifiedInputData = np.pad(
                inputData,
                tuple(
                    [(0, 0), (0, 0)] + [(self._filterWidth, self._filterWidth)] * rank
                ),
                mode="wrap",
            )

        return modifiedInputData

    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        """
        Fits the ESN so that by applying a time series out of inputData the outputData will be produced.
        """
        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimensions and rank != self.n_inputDimensions + 1:
            raise ValueError(
                "The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(
                    self.n_inputDimensions
                )
            )

        # reshape the input so that it has the shape (timeseries, time, input_dimension^n)
        if rank == self.n_inputDimensions:
            inputData = inputData.reshape(1, *inputData.shape)
            outputData = outputData.reshape(1, *outputData.shape)
        else:
            # modify rank again
            rank -= 1

        partialLength = inputData.shape[1] - transientTime
        totalLength = inputData.shape[0] * partialLength
        timeseriesCount = inputData.shape[0]

        manager = Manager()
        fitQueue = manager.Queue()

        modifiedInputData = self._embedInputData(inputData)

        if SpatioTemporalESN._isWindows():
            self.sharedNamespace.inputData = modifiedInputData
            self.sharedNamespace.outputData = outputData

        self.sharedNamespace.transientTime = transientTime

        self.sharedNamespace.partialLength = partialLength
        self.sharedNamespace.totalLength = totalLength
        self.sharedNamespace.timeseriesCount = timeseriesCount

        jobs = (
            np.stack(
                np.meshgrid(
                    *[np.arange(x) + self._filterWidth for x in inputData.shape[2:]]
                ),
                axis=rank,
            )
            .reshape(-1, rank)
            .tolist()
        )

        nJobs = len(jobs)

        self.resetState()

        iterator = FittingArrayIterator(
            modifiedInputData, outputData, jobs, self._filterWidth, self._stride, self
        )

        pool = Pool(
            processes=self._nWorkers,
            initializer=SpatioTemporalESN._init_fitProcess,
            initargs=[fitQueue, self],
        )
        pool.map_async(self._fitProcess, iterator, chunksize=self._chunkSize)

        def _processPoolWorkerResults():
            nJobsDone = 0

            if verbose > 0:
                bar = progressbar.ProgressBar(
                    max_value=nJobs, redirect_stdout=True, poll_interval=0.0001
                )
                bar.update(0)

            while nJobsDone < nJobs:
                data = fitQueue.get()

                # result of fitting
                indices, x, WOut = data
                id = self._uniqueIDFromIndices(indices)

                if WOut is None:
                    import sys

                    print(
                        "WARNING: Fit process for pixel {0} did not succeed".format(
                            indices
                        ),
                        file=sys.stderr,
                    )

                # store WOut
                if self._averageOutputWeights:
                    if WOut is not None:
                        self._WOut += WOut / np.prod(self.inputShape)
                else:
                    self._WOuts[id] = WOut

                    # store x
                self._xs[id] = x

                nJobsDone += 1
                if verbose > 0:
                    bar.update(nJobsDone)
                    if verbose > 1:
                        print(nJobsDone)

            if verbose > 0:
                bar.finish()

        _processPoolWorkerResults()

        pool.close()

    def predict(
        self, inputData, transientTime=0, update_processor=lambda x: x, verbose=0
    ):
        """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
        """

        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimensions:
            raise ValueError(
                "The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(
                    self.n_inputDimensions
                )
            )

        manager = Manager()
        predictQueue = manager.Queue()

        # workaround as predict does not support batches atm
        # add dummy dimension to let embedInputData work properly (is optimized to work for batches)
        inputData = inputData.reshape(1, *inputData.shape)
        modifiedInputData = self._embedInputData(inputData)
        modifiedInputData = modifiedInputData[0]
        inputData = inputData[0]

        if SpatioTemporalESN._isWindows():
            self.sharedNamespace.inputData = modifiedInputData

        self.transientTime = transientTime
        self.sharedNamespace.transientTime = transientTime
        predictionOutput = B.zeros(
            np.insert(self.inputShape, 0, inputData.shape[0] - transientTime)
        )

        jobs = (
            np.stack(
                np.meshgrid(
                    *[np.arange(x) + self._filterWidth for x in inputData.shape[1:]]
                ),
                axis=rank,
            )
            .reshape(-1, rank)
            .tolist()
        )
        nJobs = len(jobs)

        self.resetState()

        iterator = PredictionArrayIterator(
            modifiedInputData, jobs, self._filterWidth, self._stride, self
        )

        pool = Pool(
            processes=self._nWorkers,
            initializer=SpatioTemporalESN._init_predictProcess,
            initargs=[predictQueue, self],
        )
        pool.map_async(self._predictProcess, iterator, chunksize=self._chunkSize)

        def _processPoolWorkerResults():
            nJobsDone = 0

            if verbose > 0:
                bar = progressbar.ProgressBar(
                    max_value=nJobs, redirect_stdout=True, poll_interval=0.0001
                )
                bar.update(0)

            while nJobsDone < nJobs:
                data = predictQueue.get()
                # result of predicting
                indices, prediction, state = data
                id = self._uniqueIDFromIndices(indices)
                self._xs[id] = state
                # update the values
                predictionOutput[tuple([Ellipsis] + indices)] = prediction

                nJobsDone += 1
                if verbose > 0:
                    bar.update(nJobsDone)
                    if verbose > 1:
                        print(nJobsDone)

            if verbose > 0:
                bar.finish()

        _processPoolWorkerResults()

        pool.close()

        return predictionOutput

    def _uniqueIDFromIndices(self, indices):
        id = indices[-1]

        if len(indices) != len(self.inputShape):
            raise ValueError(
                "Shape if `indices` does not match the `inputShape` of the SpatioTemporalESN."
            )

        if len(self.inputShape) > 1:
            for i in range(0, len(self.inputShape) - 1):
                id += self.inputShape[i + 1] * indices[i]

        return id

    @staticmethod
    def _init_fitProcess(fitQueue, self):
        SpatioTemporalESN._fitProcess.fitQueue = fitQueue
        SpatioTemporalESN._fitProcess.self = self

    def _fitProcess(self, data):
        try:
            inData, outData, indices, state = data
            transientTime = self.sharedNamespace.transientTime

            partialLength = self.sharedNamespace.partialLength
            totalLength = self.sharedNamespace.totalLength
            timeseriesCount = self.sharedNamespace.timeseriesCount

            workerID = self.parallelWorkerIDs.get()
            self._x[workerID] = state

            # propagate
            X = B.empty((1 + self.n_input + self.n_reservoir, totalLength))

            for i in range(timeseriesCount):
                X[:, i * partialLength : (i + 1) * partialLength] = self.propagate(
                    inData[i],
                    transientTime=transientTime,
                    x=self._x[workerID],
                    verbose=0,
                )

            # define the target values
            Y_target = B.empty((1, totalLength))
            for i in range(timeseriesCount):
                Y_target[
                    :, i * partialLength : (i + 1) * partialLength
                ] = self.out_inverse_activation(outData[i]).T[:, transientTime:]

            # now fit
            WOut = None
            if self._solver == "pinv":
                WOut = B.dot(Y_target, B.pinv(X))

            elif self._solver == "lsqr":
                X_T = X.T
                WOut = B.dot(
                    B.dot(Y_target, X_T),
                    B.inv(
                        B.dot(X, X_T)
                        + self._regressionParameters[0]
                        * B.identity(1 + self.n_input + self.n_reservoir)
                    ),
                )

            # store the state and the output matrix of the worker
            SpatioTemporalESN._fitProcess.fitQueue.put(
                (
                    [x - self._filterWidth for x in indices],
                    self._x[workerID].copy(),
                    WOut.copy(),
                )
            )

            self.parallelWorkerIDs.put(workerID)

        except Exception as ex:
            print(ex)
            import traceback

            traceback.print_exc()

            SpatioTemporalESN._fitProcess.fitQueue.put(
                ([x - self._filterWidth for x in indices], None, None)
            )

            self.parallelWorkerIDs.put(workerID)

    @staticmethod
    def _init_predictProcess(predictQueue, self):
        SpatioTemporalESN._predictProcess.predictQueue = predictQueue
        SpatioTemporalESN._predictProcess.self = self

    def _predictProcess(self, data):
        try:
            inData, indices, state = data
            transientTime = self.sharedNamespace.transientTime
            workerID = self.parallelWorkerIDs.get()
            # get internal id
            id = self._uniqueIDFromIndices([x - self._filterWidth for x in indices])

            # now fit
            X = self.propagate(inData, transientTime=transientTime, x=state, verbose=0)

            if self._averageOutputWeights:
                WOut = self._WOut
            else:
                WOut = self._WOuts[id]

            # calculate the actual prediction
            prediction = self.out_activation(B.dot(WOut, X).T)[:, 0]

            # store the state and the output matrix of the worker
            SpatioTemporalESN._predictProcess.predictQueue.put(
                ([x - self._filterWidth for x in indices], prediction, state)
            )

            self.parallelWorkerIDs.put(workerID)
        except Exception as ex:
            print("Exception occurred in prediction thread", ex)
            import traceback

            traceback.print_exc()
