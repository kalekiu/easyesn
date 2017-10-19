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

#import dill

from multiprocess import Process, Queue, Manager, Pool, cpu_count #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#import multiprocessing
import ctypes
from multiprocessing import process

class SpatioTemporalESN(BaseESN):
    def __init__(self, inputShape, n_reservoir,
                 filterSize=1, stride=1, borderMode="unique",
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

        self._regression_parameters = regression_parameters
        self._solver = solver

        n_inputDimensions = len(inputShape)

        self._filterSize = filterSize
        self._filterWidth = filterSize // 2
        self._stride = stride

        self._n_input = int(np.power(np.ceil(filterSize / stride), n_inputDimensions))

        self.n_inputDimensions = n_inputDimensions
        self.inputShape = inputShape

        super(SpatioTemporalESN, self).__init__(n_input=self._n_input, n_reservoir=n_reservoir, n_output=1, spectralRadius=spectralRadius,
                                  noiseLevel=noiseLevel, inputScaling=inputScaling, leakingRate=leakingRate, sparseness=sparseness,
                                  random_seed=random_seed, out_activation=out_activation, out_inverse_activation=out_inverse_activation,
                                  weight_generation=weight_generation, bias=bias, output_bias=output_bias, outputInputScaling=outputInputScaling,
                                  input_density=input_density, activation=activation)

        manager = Manager()
        self._sharedNamespace = manager.Namespace()
        self._fitQueue = manager.Queue()

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
    
    def resetState(self, index=None):
        if index is None:
             self._x = B.zeros((self._nWorkers, self.n_reservoir, 1))
        else:
            self._x[index] = B.zeros((self.n_reservoir, 1))

    """
        Fits the ESN so that by applying a time series out of inputData the outputData will be produced.
    """
    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimensions:
            raise ValueError("The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(self.n_inputDimensions))

        self._sharedNamespace.inputData = inputData
        self._sharedNamespace.outputData = outputData
        self._sharedNamespace.transientTime = transientTime
       
        jobs = np.stack(np.meshgrid(*[np.arange(self._filterWidth, x-self._filterWidth) for x in inputData.shape[1:]]), axis=rank).reshape(-1, rank).tolist()
        nJobs = len(jobs)

        self._sharedNamespace.WOuts = B.empty((nJobs, 1+self.n_reservoir+1, 1))

        self._nWorkers = np.max((cpu_count()-1, 1))
        pool = Pool(processes=self._nWorkers)

        self._fitWorkerIDs = list(range(self._nWorkers))

        self.resetState()

        processProcessResultsThread = Process(target=self._processPoolWorkerResults, args=(nJobs, verbose))
        processProcessResultsThread.start()


        results = pool.map(self._fitProcess, jobs)
        pool.close()

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
       pass

    def _processPoolWorkerResults(self, nJobs, verbose):
        nJobsDone = 0
        
        if verbose > 0:
            bar = progressbar.ProgressBar(max_value=nJobs, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        while nJobsDone < nJobs:
            indices, x, WOut = self._fitQueue.get()
            
            #store x and WOut

            nJobsDone += 1
            if verbose > 0:
                bar.update(nJobsDone)
                if verbose > 1:
                    print(nJobsDone)

        
        if verbose > 0:
            bar.finish()

        if self._averageOutputWeights:
            print("calculating average output matrix...")
            self._WOut = B.mean(self._WOuts, axis=0)

    def _fitProcess(self, indices):
        inputData = self._sharedNamespace.inputData
        outputData = self._sharedNamespace.outputData
        transientTime = self._sharedNamespace.transientTime
       
        y, x = indices
        workerID = self._fitWorkerIDs[0]
        self._fitWorkerIDs.pop(0)
        #create patchedInputData

        #treat the frame pixels in a special way
        inData = inputData[:, y-self._filterWidth:y+self._filterWidth+1, x-self._filterWidth:x+self._filterWidth+1][:, ::self._stride, ::self._stride].reshape(len(inputData), -1)

        #create target output series
        outData = outputData[:, y, x].reshape(-1, 1)

        #now fit
        X = self.propagate(inData, transientTime, x=self._x[workerID], verbose=0)

        #define the target values
        Y_target = self.out_inverse_activation(outData).T[:, transientTime:]

        X_T = X.T
        WOut = B.dot(B.dot(Y_target, X_T),B.inv(B.dot(X, X_T) + self._regression_parameters[0]*B.identity(1+self.n_input+self.n_reservoir)))

        #calculate the training prediction now
        trainingPrediction = self.out_activation(B.dot(WOut, X).T)
            
        #store the state and the output matrix of the worker
        self._fitQueue.put(indices, self._x[workerID].copy(), WOut.copy())

        self._fitWorkerIDs.append(workerID)