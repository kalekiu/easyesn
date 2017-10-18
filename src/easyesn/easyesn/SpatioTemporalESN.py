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

import multiprocessing


class SpatioTemporalESN(BaseESN):
    def __init__(self, inputShape, n_reservoir,
                 filterSize=1, stride=1, borderMode="unique"
                 spectralRadius=1.0, noiseLevel=0.0, inputScaling=None,
                 leakingRate=1.0, sparseness=0.2, random_seed=None, averageOutputWeights=True,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weight_generation='naive', bias=1.0, output_bias=1.0,
                 outputInputScaling=1.0, input_density=1.0, solver='pinv', regression_parameters={}, activation = B.tanh):

        self._averageOutputWeights = averageOutputWeights
        if averageOutputWeights and solver != "lsqr":
            raise ValueError("`averageOutputWeights` can only be set to `True` when `solver` is set to `lsqr` (Ridge Regression)")

        self._borderMode = borderMode
        if not borderMode in ["mirror", "padding"]:
            raise ValueError("`borderMode` must be set to one of the following values: `mirror` or `padding`.")

        n_inputDimensions = len(inputShape)

        self._filterSize = filterSize
        self._stride = stride

        self._n_input = int(np.pow(np.ceil(filterSize / stride), n_inputDimensions))

        self.n_inputDimensions = n_inputDimensions
        self.inputShape = inputShape

        self._spectralRadius = spectralRadius
        self._noiseLevel = noiseLevel
        self._inputScaling = inputScaling
        self._leakingRate = leakingRate
        self._sparseness = sparseness
        self._random_seed = random_seed
        self._out_activation = out_activation
        self._out_inverse_activation = out_inverse_activation
        self._weight_generation = weight_generation
        self._bias = bias
        self._output_bias = output_bias
        self._outputInputScaling = outputInputScaling
        self._input_density = input_density
        self._solver = solver
        self._regression_parameters = regression_parameters
        self._activation = activation

        self._manager = multiprocessing.Manager()
        self._sharedNamespace = self._manager.Namespace()

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
        
    def resetState(self):
        self._x = B.zeros((self._nWorkers, self.n_reservoir, 1))

    
    def resetState(self, index):
        self._x[index] = B.zeros((self.n_reservoir, 1))

    """
        Fits the ESN so that by applying a time series out of inputData the outputData will be produced.
    """
    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        if len(inputData.shape) == 1
            inputData = inputData.reshape(1, -1)

        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimension:
            raise ValueError("The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(self.n_inputDimension)).

        self._sharedNamespace.inputData = inputData
        self._sharedNamespace.outputData = outputData

        jobs = np.stack(np.meshgrid(*[np.arange(x) for x in inputData.shape[:-1]]), axis=len(rank))
        nJobs = np.prod(inputData.shape[:-1])

        self._sharedNamespace._WOuts = B.empty(nJobs, 1+self.n_reservoir+1, self.n_out)

        def processThreadResults(queue, nJobs, verbose):
            nJobsDone = 0

            while nJobsDone < nJobs:
                newData = queue.get()

                nJobsDone += 1

                print(nJobsDone)


        def fitProcessInit(q):
            fitProcess.q = q

        def fitProcess(indices):
            workerIndex, y, x = indices
            #create patchedInputData
            inData = inputData[y-self._filterSize:y+self._filterSize::self._stride, x-self._filterSize:x+self._filterSize::self._stride, :]

            #create target output series
            outData = outputData[y, x]

            #now fit
            X = self.propagate(inputData, transientTime, verbose)

            #define the target values
            Y_target = self.out_inverse_activation(outData).T[:, transientTime:]

            X_T = X.T
            WOut = B.dot(B.dot(Y_target, X_T),B.inv(B.dot(self._X, X_T) + self._regression_parameters[0]*B.identity(1+self.n_input+self.n_reservoir)))

            #calculate the training prediction now
            trainingPrediction = self.out_activation(B.dot(WOut, X).T)
            
            #store the state and the output matrix of the worker
            fitProcess.q.put(indices, self.x[workerIndex].copy(), WOut.copy())

        queue = Queue()
        self._nWorkers = np.max(multiprocessing.cpu_count()-1, 1)
        pool = Pool(processes=self._nWorkers, initializer=fitProcessInit, initargs=[queue,] )

        self.resetState()

        processProcessResultsThread = Process(target=processThreadResults, args=(queue, nJobs, verbose))
        processProcessResultsThread.start()

        results = pool.map(fitProcess, jobs)
        pool.close()

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
       
