"""
    Performs basic grid search for ESNs in which the parameter space will be searched in discrete steps.
"""

import numpy as np
import itertools
import operator
import progressbar
from .. import helper as hlp
import matplotlib.pyplot as plt

"""
    Performs basic grid search for ESNs in which the parameter space will be searched in discrete steps.
"""
class GridSearchOptimizer:
    def __init__(self, param_grid, fixed_params, esnType):
        self.esnType = esnType
        self.param_grid = param_grid
        self.fixed_params = fixed_params


    """
        Fits an ESN for each of the wanted hyperparameters and predicts the output.
        The best results parameters will be stores in _best_params.
    """
    def fit(self, trainingInput, trainingOutput, validationInput, validationOutput, verbose=1):
        #calculate the length of all permutations of the hyperparameters
        def enumerate_params():
            keys, values = zip(*self.param_grid.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))
        length = sum(1 for x in enumerate_params())

        if verbose > 0:
            #initialize the progressbar to indicate the progress
            bar = progressbar.ProgressBar(max_value=length, redirect_stdout=True)

        #store the results here
        results = []

        for index, params in enumerate(enumerate_params()):
            #create and fit the ESN
            esn = self.esnType(**params, **self.fixed_params)
            trainingAccuricy = esn.fit(trainingInput, trainingOutput)

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

            results.append((validationMSE, trainingAccuricy, params))
            
            if verbose > 0:
                bar.update(index)

            #print the currently best result every printfreq step
            if verbose > 1:
                res = min(results, key=operator.itemgetter(0))
                print("Current best parameters: \t: " + str(res))

        if verbose > 0:
            bar.finish()

        #determine the best parameters by minimizing the error
        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[2]
        self._best_mse = res[0]

        return results

    def plotGrid(self, grid, title, fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight=None):
        print(gridHeight)
        plotGrid = np.copy(grid)
        if gridHeight is not None:
            maxValue = np.min(grid) + np.min(grid) * gridHeight
            plotGrid[np.where(plotGrid > maxValue)] = maxValue
            plotGrid[np.where(np.isnan(plotGrid))] = maxValue
        fig, ax = plt.subplots()
        mat = plt.imshow(plotGrid, vmin=np.min(plotGrid), vmax=np.max(plotGrid),
                         extent=[fromParam1, tillParam1, tillParam2, fromParam2])
        clb = fig.colorbar(mat)
        plt.title(title)
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.show()


    def plotErrorSurface(self, esn, N, transientTime, trainInputs, trainTargets, validationInputs, validationTargets, paramDic, gridHeight = None, verbose=1):
        def setParameter(esn, parameter, value):
            if parameter == "Spectral radius":
                esn.setSpectralRadius(value)
            elif parameter == "Leaking rate":
                esn.setLeakingRate(value)
            elif parameter == "Input scaling":
                esn.setInputScaling(value)
            else:
                raise ValueError("Hyperparameter {0} does not exist.".format(parameter))


        if len(list(paramDic.keys())) != 2:
            raise ValueError("Can only plot error surface of exactly 2 Hyperparameter")

        if verbose > 0:
            # initialize the progressbar to indicate the progress
            bar = progressbar.ProgressBar(max_value=N*N-1, redirect_stdout=True)

        param1 = list(paramDic.keys())[0]
        fromParam1, tillParam1 = list(paramDic.values())[0]

        param2 = list(paramDic.keys())[1]
        fromParam2, tillParam2 = list(paramDic.values())[1]



        trainErrorGrid = np.zeros((N, N))
        validationErrorGrid = np.zeros((N, N))
        widthParam1 = (tillParam1 - fromParam1) / N
        widthParam2 = (tillParam2 - fromParam2) / N
        for i in range(N):
            p1 = i * widthParam1 + fromParam1
            setParameter(esn, param1, p1)
            for j in range(N):
                p2 = j * widthParam2 + fromParam2
                setParameter(esn, param2, p2)

                trainErrorGrid[i, j] = esn.fit(trainInputs, trainTargets, transientTime=transientTime)
                validationErrorGrid[i, j] = hlp.loss(esn.predict(validationInputs), validationTargets)

                if verbose > 0:
                    bar.update(i*N + j)

        self.plotGrid(trainErrorGrid, "Train error", fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight)

        self.plotGrid(validationErrorGrid, "Validation error", fromParam1, tillParam1, fromParam2, tillParam2, param1, param2, gridHeight)



        return trainErrorGrid, validationErrorGrid
