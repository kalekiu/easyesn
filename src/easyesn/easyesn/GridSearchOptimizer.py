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
    def fit(self, trainingInput, trainingOutput, testingDataSequence, output_postprocessor = lambda x: x, printfreq=1):
        #calculate the length of all permutations of the hyperparameters
        def enumerate_params():
            keys, values = zip(*self.param_grid.items())
            for row in itertools.product(*values):
                yield dict(zip(keys, row))
        length = sum(1 for x in enumerate_params())

        #iniitialize the progressbar to indicate the progress
        bar = progressbar.ProgressBar(max_value=length, redirect_stdout=True)

        #store the results here
        results = []

        suc = 0
        for params in enumerate_params():
            #create and fit the ESN
            esn = self.esnType(**params, **self.fixed_params)
            training_acc = esn.fit(trainingInput, trainingOutput)

            current_state = esn._x

            #evaluate the ESN
            test_mse = []
            for (testInput, testOutput) in testingDataSequence:
                esn._x = current_state
                out_pred = output_postprocessor(esn.predict(testInput))
                test_mse.append(np.mean((testOutput - out_pred)**2))

            test_mse = np.mean(test_mse)

            results.append((test_mse, training_acc, params))

            suc += 1
            bar.update(suc)

            #print the currently best result every printfreq step
            if (suc % printfreq == 0):
                res = min(results, key=operator.itemgetter(0))
                print("\t: " + str(res))

        #determine the best parameters by minimizing the error
        res = min(results, key=operator.itemgetter(0))

        self._best_params = res[2]
        self._best_mse = res[0]

        return results

