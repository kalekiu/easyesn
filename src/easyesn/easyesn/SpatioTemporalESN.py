"""
    Implementation of the general ESN model.
"""

import numpy as np
import numpy.random as rnd
from .PredictionESN import PredictionESN

from . import backend as B

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import progressbar

import multiprocessing


class SpatioTemporalESN(object):
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

    def

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


        def fitProcessInit(q):
            fitProcess.q = q

        def fitProcess(indices):
            #create patchedInputData

            #create target output series

            esn.fit(inputData=patchedInputData, outputData=targetOutputData, transientTime=transientTime, verbose=0)
            fitProcess.q.put(indices, self.x, self._WOut)

        queue = Queue()
        pool = Pool(processes=nJobs, initializer=fitProcessInit, initargs=[queue,] )

        processProcessResultsThread = Process(target=processThreadResults, args=(queue, nJobs, verbose))
        processProcessResultsThread.start()

        results = pool.map(fitProcess, jobs)
        pool.close()

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
       
