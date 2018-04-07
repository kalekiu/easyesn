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
import sys

#import dill

from multiprocess import Process, Queue, Manager, Pool, cpu_count #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
#import multiprocessing
import ctypes
from multiprocessing import process

class SpatioTemporalESN(BaseESN):
    def __init__(self, inputShape, n_reservoir,
                 filterSize=1, stride=1, borderMode="mirror", nWorkers="auto",
                 spectralRadius=1.0, noiseLevel=0.0, inputScaling=None,
                 leakingRate=1.0, reservoirDensity=0.2, randomSeed=None, averageOutputWeights=True,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weightGeneration='naive', bias=1.0, outputBias=1.0,
                 outputInputScaling=1.0, inputDensity=1.0, solver='pinv', regressionParameters={}, activation = B.tanh, activationDerivation=lambda x: 1.0/B.cosh(x)**2):
       
        self._averageOutputWeights = averageOutputWeights
        if averageOutputWeights and solver != "lsqr":
            raise ValueError("`averageOutputWeights` can only be set to `True` when `solver` is set to `lsqr` (Ridge Regression)")

        self._borderMode = borderMode
        if not borderMode in ["mirror", "padding", "edge", "wrap"]:
            raise ValueError("`borderMode` must be set to one of the following values: `mirror`, `padding`, `edge` or `wrap`.")

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
            self._WOuts = B.empty((np.prod(inputShape), 1, self._n_input+n_reservoir+1))
            self._WOut = None
        else:
            self._WOuts = None
            self._WOut = B.zeros((1, self._n_input+n_reservoir+1))
        self._xs = B.empty((np.prod(inputShape), n_reservoir, 1))

        
        if nWorkers=="auto":
            self._nWorkers = np.max((cpu_count()-1, 1))
        else:
            self._nWorkers = nWorkers
            

        manager = Manager()
        self.sharedNamespace = manager.Namespace()
        if hasattr(self, "fitWorkerID") == False or self.parallelWorkerIDs is None:
            self.parallelWorkerIDs = manager.Queue()
            for i in range(self._nWorkers):
                self.parallelWorkerIDs.put((i))

        super(SpatioTemporalESN, self).__init__(n_input=self._n_input, n_reservoir=n_reservoir, n_output=1, spectralRadius=spectralRadius,
                                  noiseLevel=noiseLevel, inputScaling=inputScaling, leakingRate=leakingRate, reservoirDensity=reservoirDensity,
                                  randomSeed=randomSeed, out_activation=out_activation, out_inverse_activation=out_inverse_activation,
                                  weightGeneration=weightGeneration, bias=bias, outputBias=outputBias, outputInputScaling=outputInputScaling,
                                  inputDensity=inputDensity, activation=activation, activationDerivation=activationDerivation)

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
        return hasattr(sys, 'getwindowsversion')

    def resetState(self, index=None):
        if index is None:
             self._x = B.zeros((self._nWorkers, self.n_reservoir, 1))
        else:
            self._x[index] = B.zeros((self.n_reservoir, 1))

    def _embedInputData(self, inputData):
        rank = len(inputData.shape) - 2

        if self._borderMode == "mirror":
            modifiedInputData = np.pad(inputData, tuple([(0,0), (0,0)] + [(self._filterWidth, self._filterWidth)]*rank), mode="symmetric")
        elif self._borderMode == "padding":
            modifiedInputData = np.pad(inputData, tuple([(0,0), (0,0)] + [(self._filterWidth, self._filterWidth)]*rank), mode="constant", constant_values=0)
        elif self._borderMode == "edge":
            modifiedInputData = np.pad(inputData, tuple([(0,0), (0,0)] + [(self._filterWidth, self._filterWidth)]*rank), mode="edge")
        elif self._borderMode == "wrap":
            modifiedInputData = np.pad(inputData, tuple([(0,0), (0,0)] + [(self._filterWidth, self._filterWidth)]*rank), mode="wrap")

        return modifiedInputData

    """
        Fits the ESN so that by applying a time series out of inputData the outputData will be produced.
    """
    def fit(self, inputData, outputData, transientTime=0, verbose=0):
        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimensions and rank != self.n_inputDimensions + 1:
            raise ValueError("The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(self.n_inputDimensions))

        # reshape the input so that it has the shape (timeseries, time, input_dimension^n)
        if rank == self.n_inputDimensions:
            inputData = inputData.reshape(1, *inputData.shape)
            outputData = outputData.reshape(1, *outputData.shape)
        else:
            # modify rank again
            rank -= 1

        partialLength = (inputData.shape[1]-transientTime)
        totalLength = inputData.shape[0] * partialLength
        timeseriesCount = inputData.shape[0] 

        manager = Manager()
        fitQueue = manager.Queue()

        modifiedInputData = self._embedInputData(inputData)
        
        if SpatioTemporalESN._isWindows():
            self.sharedNamespace.inputData = modifiedInputData
            self.sharedNamespace.outputData = outputData
        else:
            self._inputData = modifiedInputData
            self._outputData = outputData
        
        self.sharedNamespace.transientTime = transientTime

        self.sharedNamespace.partialLength = partialLength
        self.sharedNamespace.totalLength = totalLength
        self.sharedNamespace.timeseriesCount = timeseriesCount

        self.sharedNamespace.WOut = self._WOut
        self.sharedNamespace.WOuts = self._WOuts
        self.sharedNamespace.xs = self._xs
     
        jobs = np.stack(np.meshgrid(*[np.arange(x)+self._filterWidth for x in inputData.shape[2:]]), axis=rank).reshape(-1, rank).tolist()
        nJobs = len(jobs)

        self.sharedNamespace.WOuts = self._WOuts

        self.resetState()
        
        pool = Pool(processes=self._nWorkers, initializer=SpatioTemporalESN._init_fitProcess, initargs=[fitQueue, self])  
        pool.map_async(self._fitProcess, jobs)

        def _processPoolWorkerResults():
            nJobsDone = 0
        
            if verbose > 0:
                bar = progressbar.ProgressBar(max_value=nJobs, redirect_stdout=True, poll_interval=0.0001)
                bar.update(0)

            while nJobsDone < nJobs:
                data = fitQueue.get()
            
                #result of fitting
                indices, x, WOut = data
                id = self._uniqueIDFromIndices(indices)
                
                if WOut is None:
                    import sys
                    print("WARNING: Fit process for pixel {0} did not suceed".format(indices), file=sys.stderr)

                #store WOut
                if self._averageOutputWeights:
                    if WOut is not None:
                        self._WOut += WOut/np.prod(self.inputShape)
                else:
                    self._WOuts[id] = WOut 

                #store x
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

        self.sharedNamespace.WOut = self._WOut
        self.sharedNamespace.WOuts = self._WOuts
        self.sharedNamespace.xs = self._xs

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, transientTime=0, update_processor=lambda x:x, verbose=0):
        rank = len(inputData.shape) - 1

        if rank != self.n_inputDimensions:
            raise ValueError("The `inputData` does not have a suitable shape. It has to have {0} spatial dimensions and 1 temporal dimension.".format(self.n_inputDimensions))

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
        else:
            self._inputData = modifiedInputData
        self.transientTime = transientTime
        predictionOutput = B.zeros(np.insert(self.inputShape, 0, inputData.shape[0]-transientTime))
       
        jobs = np.stack(np.meshgrid(*[np.arange(x)+self._filterWidth for x in inputData.shape[1:]]), axis=rank).reshape(-1, rank).tolist()
        nJobs = len(jobs)

        self.resetState()

        pool = Pool(processes=self._nWorkers, initializer=SpatioTemporalESN._init_predictProcess, initargs=[predictQueue, self])  
        pool.map_async(self._predictProcess, jobs)
        
        def _processPoolWorkerResults():
            nJobsDone = 0
        
            if verbose > 0:
                bar = progressbar.ProgressBar(max_value=nJobs, redirect_stdout=True, poll_interval=0.0001)
                bar.update(0)

            while nJobsDone < nJobs:
                data = predictQueue.get()
                #result of predicting
                indices, prediction = data
                #update the values
                predictionOutput[tuple([Ellipsis] +  indices)] = prediction
                         
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
            raise ValueError("Shape if `indices` does not match the `inputShape` of the SpatioTemporalESN.")

        if len(self.inputShape) > 1:
            for i in range(0, len(self.inputShape)-1):
                id += self.inputShape[i+1]*indices[i]

        return id

    @staticmethod
    def _init_fitProcess(fitQueue, self):
        SpatioTemporalESN._fitProcess.fitQueue = fitQueue
        SpatioTemporalESN._fitProcess.self = self

    def _fitProcess(self, indices):
        try:
            if SpatioTemporalESN._isWindows():
                inputData = self.sharedNamespace.inputData
                outputData = self.sharedNamespace.outputData
            else:
                inputData = self._inputData
                outputData = self._outputData
            transientTime = self.sharedNamespace.transientTime

            partialLength = self.sharedNamespace.partialLength
            totalLength = self.sharedNamespace.totalLength
            timeseriesCount = self.sharedNamespace.timeseriesCount

            y, x = indices

            #print(id(SpatioTemporalESN._fitProcess.sharedNamespace.parallelWorkerIDs))
            workerID = self.parallelWorkerIDs.get()
            #self.sharedNamespace.parallelWorkerIDs = self.sharedNamespace.parallelWorkerIDs

            #create patchedInputData
            #treat the frame pixels in a special way
            inData = inputData[:, :, y-self._filterWidth : y+self._filterWidth+1, x-self._filterWidth : x+self._filterWidth+1][:, ::self._stride, ::self._stride].reshape(len(inputData), inputData.shape[1], -1)
            #create target output series
            outData = outputData[:, :, y-self._filterWidth, x-self._filterWidth].reshape(len(outputData), -1, 1)

            # propagate
            X = B.empty((1 + self.n_input + self.n_reservoir, totalLength))

            for i in range(timeseriesCount):
                X[:, i*partialLength:(i+1)*partialLength] = self.propagate(inData[i], transientTime=transientTime, x=self._x[workerID], verbose=0)

            #define the target values
            Y_target = B.empty((1, totalLength))
            for i in range(timeseriesCount):
                Y_target[:, i*partialLength:(i+1)*partialLength] = self.out_inverse_activation(outData[i]).T[:, transientTime:]

            # now fit
            WOut = None
            if (self._solver == "pinv"):
                WOut = B.dot(Y_target, B.pinv(X))

            elif (self._solver == "lsqr"):
                X_T = X.T
                WOut = B.dot(B.dot(Y_target, X_T),B.inv(B.dot(X, X_T) + self._regressionParameters[0]*B.identity(1+self.n_input+self.n_reservoir)))

            # calculate the training prediction now
            # trainingPrediction = self.out_activation(B.dot(WOut, X).T)
        
            #store the state and the output matrix of the worker
            SpatioTemporalESN._fitProcess.fitQueue.put(([x-self._filterWidth for x in indices], self._x[workerID].copy(), WOut.copy()))
        
            self.parallelWorkerIDs.put(workerID)

        except Exception as ex:
            print(ex)

            SpatioTemporalESN._fitProcess.fitQueue.put(([x-self._filterWidth for x in indices], None, None))

            self.parallelWorkerIDs.put(workerID)

    @staticmethod
    def _init_predictProcess(predictQueue, self):
        SpatioTemporalESN._predictProcess.predictQueue = predictQueue
        SpatioTemporalESN._predictProcess.self = self

    def _predictProcess(self, indices):
        try:
            if SpatioTemporalESN._isWindows():
                inputData = self.sharedNamespace.inputData
            else:
                inputData = self._inputData
            transientTime = self.sharedNamespace.transientTime

            y, x = indices
            workerID = self.parallelWorkerIDs.get()
            #get internal id
    
            id = self._uniqueIDFromIndices([x-self._filterWidth for x in indices])
                
            #create patchedInputData
            #treat the frame pixels in a special way
            inData = inputData[:, y-self._filterWidth:y+self._filterWidth+1, x-self._filterWidth:x+self._filterWidth+1][:, ::self._stride, ::self._stride].reshape(len(inputData), -1)

            self._x[workerID] = self.sharedNamespace.xs[id]
        
            #now fit
            X = self.propagate(inData, transientTime=transientTime, x=self._x[workerID], verbose=0)
            self.sharedNamespace.xs[id] = self._x[workerID]
            
            if self._averageOutputWeights:
                WOut = self._WOut
            else:
                WOut = self._WOuts[id]

            #calculate the actual prediction
            prediction = self.out_activation(B.dot(WOut, X).T)[:, 0]
                
            #store the state and the output matrix of the worker
            SpatioTemporalESN._predictProcess.predictQueue.put(([x-self._filterWidth for x in indices], prediction))
            
            self.parallelWorkerIDs.put(workerID)
        except Exception as ex:
            print(x,y)
            print(ex)

            #SpatioTemporalESN._predictProcess.predictQueue.put(([x-self._filterWidth for x in indices], None))
            #self.parallelWorkerIDs.put(workerID)