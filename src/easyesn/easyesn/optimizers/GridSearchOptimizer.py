"""
    Performs basic grid search for ESNs in which the parameter space will be searched in discrete steps.
"""

import numpy as np
import itertools
import operator
import progressbar
from .. import helper as hlp

from multiprocessing import Process, Queue, Manager, Pool #we require Pathos version >=0.2.6. Otherwise we will get an "EOFError: Ran out of input" exception
import multiprocessing

class GridSearchOptimizer:
    """
    Performs basic grid search for ESNs in which the parameter space will be searched in discrete steps.
    """

    def __init__(self, esnType, parametersDictionary={}, fixedParametersDictionary={}):
        self.esnType = esnType
        self.parametersDictionary = parametersDictionary
        self.fixedParametersDictionary = fixedParametersDictionary

    """
        Processes the async. results of the _get_score methods and indicates the progress to the user.
    """
    @staticmethod
    def _processThreadResults(q, numberOfResults, verbose):
        #no output wanted, so return now
        if verbose == 0:
            return

        #initialize the progressbar to indicate the progress
        metrics_widget = progressbar.widgets.FormatCustomText(' Loss:\t%(loss).2E', {'loss': np.nan})
        bar = progressbar.ProgressBar(max_value=numberOfResults, redirect_stdout=True, widgets=[metrics_widget])
        bar.widgets = bar.default_widgets() + [metrics_widget]
        bar.update(0)


        finishedResults = 0
        validationMSE = np.inf

        while True:
            #leave this method only if all results have been fetched
            if finishedResults == numberOfResults:
                break

            #fetch new data
            newValidationMSE, _, _ = q.get()
            validationMSE = min(validationMSE, newValidationMSE)
            finishedResults += 1

            metrics_widget.update_mapping(loss=validationMSE)

            #show progress
            bar.update(finishedResults)

        bar.finish()

    """
        Fits an ESN with one specified set of hyperparameters, evaluates and returns its performance.
    """
    @staticmethod
    def _getScore(data):
        params, fixed_params, trainingInput, trainingOutput, validationInput, validationOutput, transientTime, esnType = data

        try:
            esn = esnType(**params, **fixed_params)

            trainingAccuracy = esn.fit(trainingInput, trainingOutput, transientTime=transientTime)

            current_state = esn._x

            #evaluate the ESN
            validationMSEs = []

            #check whether only one validation sequence is ought to be checked or if the esn has to be validated on multiple sequences
            if len(validationOutput.shape) == len(trainingInput.shape) + 1:
                for n in range(validationOutput.shape[0]):
                    esn._x = current_state
                    outputPrediction = esn.predict(validationInput[n])
                    validationMSEs.append(np.mean((validationOutput[n] - outputPrediction)**2))
            else:
                 esn._x = current_state
                 outputPrediction = esn.predict(validationInput)
                 validationMSEs.append(np.mean((validationOutput - outputPrediction)**2))

            validationMSE = np.mean(validationMSEs)

            dat = (validationMSE, trainingAccuracy, params)
        except:
            import sys, traceback
            print("Unexpected error:", sys.exc_info()[0])
            print(traceback.format_exc())

            dat = (np.nan, np.nan, params)


        GridSearchOptimizer._getScore.q.put(dat)

        return dat

    """
        Initializes the queue object of the _get_score method.
    """
    @staticmethod
    def _getScoreInit(q):
        GridSearchOptimizer._getScore.q = q

    """
        Fits an ESN for each of the wanted hyperparameters and predicts the output.
        The best results parameters will be stores in _best_params.
    """
    def fit_parallel(self, trainingInput, trainingOutput, validationInput, validationOutput, transientTime, verbose=1, n_jobs=1):
        #calculate the length of all permutations of the hyperparameters
        #create the jobs
        def enumerate_params():
            keys, values = zip(*self.parametersDictionary.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))
        jobs = []
        for x in enumerate_params():
            jobs.append((x, self.fixedParametersDictionary, trainingInput, trainingOutput, validationInput, validationOutput, transientTime, self.esnType))

        queue = Queue()
        pool = Pool(processes=n_jobs, initializer=GridSearchOptimizer._getScoreInit, initargs=[queue,] )

        processProcessResultsThread = Process(target=GridSearchOptimizer._processThreadResults, args=(queue, len(jobs), verbose))

        processProcessResultsThread.start()
        results = pool.map(GridSearchOptimizer._getScore, jobs)
        pool.close()

        #determine the best parameters by minimizing the error
        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[2]
        self._best_mse = res[0]

        return results


    def fit(self, trainingInput, trainingOutput, validationInput, validationOutput, transientTime, verbose=1):
        """
        Fits an ESN for each of the wanted hyperparameters and predicts the output.
        The best results parameters will be stores in _best_params.
        """

        #calculate the length of all permutations of the hyperparameters
        def enumerate_params():
            keys, values = zip(*self.parametersDictionary.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))
        length = sum(1 for x in enumerate_params())

        if verbose > 0:
            #initialize the progressbar to indicate the progress
            metrics_widget = progressbar.widgets.FormatCustomText(' Loss:\t%(loss).2E', {'loss': np.nan})
            bar = progressbar.ProgressBar(max_value=length, redirect_stdout=True, widgets=[metrics_widget])
            bar.widgets = bar.default_widgets() + [metrics_widget]

        #store the results here
        results = []

        for index, params in enumerate(enumerate_params()):
            #create and fit the ESN
            esn = self.esnType(**params, **self.fixedParametersDictionary)
            trainingAccuracy = esn.fit(trainingInput, trainingOutput, transientTime=transientTime)

            current_state = esn._x

            #evaluate the ESN
            validationMSEs = []

            #check whether only one validation sequence is ought to be checked or if the esn has to be validated on multiple sequences
            if len(validationOutput.shape) == len(trainingInput.shape) + 1:
                for n in range(validationOutput.shape[0]):
                    esn._x = current_state
                    outputPrediction = esn.predict(validationInput[n])
                    validationMSEs.append(np.mean((validationOutput[n] - outputPrediction)**2))
            else:
                 esn._x = current_state
                 outputPrediction = esn.predict(validationInput)
                 validationMSEs.append(np.mean((validationOutput - outputPrediction)**2))

            validationMSE = np.mean(validationMSEs)

            results.append((validationMSE, trainingAccuracy, params))
            
            if verbose > 0:
                bar.update(index)

            #print the currently best result every printfreq step
            if verbose > 1:
                res = min(results, key=operator.itemgetter(0))
                print("Current best parameters: \t: " + str(res))

            if verbose > 0:
                metrics_widget.update_mapping(loss=min(results, key=operator.itemgetter(0))[0])

        if verbose > 0:
            bar.finish()

        #determine the best parameters by minimizing the error
        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[2]
        self._best_mse = res[0]

        return results

    def createDenseHyperparameterGrid(self, parameterGridDictionary):
        """
        Creates a dense grid of hyperparameters for the settings specified inside `parameterGridDictionary`.

        Args:
            parameterGridDictionary (dictionary): Dictionary with strings as keys, which correspond to the name of a hyperparameter of an ESN.
                                                  The values are tuples (start, end, numberOfSteps) which describe the sampling of each parameter's space.
        """
        for key, values in parameterGridDictionary:
            fromParam, tillParam, steps = values
            parameterGridDictionary[key] = np.linspace(fromParam, tillParam, steps)

        self.parametersDictionary = parameterGridDictionary

    @staticmethod
    def plotErrorSurface(esn, N, transientTime, trainInputs, trainTargets, validationInputs, validationTargets, paramDic, gridHeight = None, verbose=1):
        """

        Args:

        Returns:
            trainErrorGrid ():
            validationErrorGrid ():
        """

        if len(list(paramDic.keys())) != 2:
            raise ValueError("Can only plot error surface of exactly 2 hyperparameters")

        if verbose > 0:
            # initialize the progressbar to indicate the progress
            bar = progressbar.ProgressBar(max_value=N*N-1, redirect_stdout=True)

        param1 = list(paramDic.keys())[0]
        fromParam1, tillParam1 = list(paramDic.values())[0]

        param2 = list(paramDic.keys())[1]
        fromParam2, tillParam2 = list(paramDic.values())[1]

        def setParameter(parameter, value):
            if parameter == "spectralRadius":
                esn.setSpectralRadius(value)
            elif parameter == "leakingRate":
                esn.setLeakingRate(value)
            elif parameter == "inputScaling":
                esn.setInputScaling(value)
            else:
                raise ValueError("Hyperparameter {0} does not exist. Choose from either spectralRadius, leakingRate, inputScaling".format( parameter))

        def plotGrid(grid, title, fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight=None):
            plotGrid = np.copy(grid)
            if gridHeight is not None:
                maxValue = np.min(grid) + np.min(grid) * gridHeight
                plotGrid[np.where(plotGrid > maxValue)] = maxValue
                plotGrid[np.where(np.isnan(plotGrid))] = maxValue

            import matplotlib.pyplot as plt

            fig = plt.figure()
            mat = plt.imshow(plotGrid, vmin=np.min(plotGrid), vmax=np.max(plotGrid),
                             extent=[fromParam1, tillParam1, tillParam2, fromParam2])
            clb = fig.colorbar(mat)
            plt.title(title)
            plt.xlabel(param1)
            plt.ylabel(param2)
            return fig

        trainErrorGrid = np.zeros((N, N))
        validationErrorGrid = np.zeros((N, N))
        widthParam1 = (tillParam1 - fromParam1) / N
        widthParam2 = (tillParam2 - fromParam2) / N
        for i in range(N):
            p1 = i * widthParam1 + fromParam1
            setParameter(param1, p1)
            for j in range(N):
                p2 = j * widthParam2 + fromParam2
                setParameter(param2, p2)

                trainErrorGrid[i, j] = esn.fit(trainInputs, trainTargets, transientTime=transientTime)
                validationErrorGrid[i, j] = hlp.loss(esn.predict(validationInputs), validationTargets)

                if verbose > 0:
                    bar.update(i*N + j)

        if verbose > 0:
            bar.finish()

        trainFigure = plotGrid(trainErrorGrid, "Train error", fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight)
        validationFigure = plotGrid(validationErrorGrid, "Validation error", fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight)

        return trainFigure, validationFigure, trainErrorGrid, validationErrorGrid
