"""
    Performs basic grid search for ESNs in which the parameter space will be searched in discrete steps.
"""

import numpy as np
import itertools
import operator
import progressbar

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



