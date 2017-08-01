"""
    Implementation of the general ESN model.
"""

import numpy as np
import numpy.random as rnd
from BaseESN import BaseESN

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
import progressbar


class ESN(BaseESN):
    def __init__(self, n_input, n_reservoir, n_output,
                 spectral_radius=1.0, noise_level=0.01, input_scaling=None,
                 leak_rate=1.0, sparseness=0.2, random_seed=None,
                 out_activation=lambda x: x, out_inverse_activation=lambda x: x,
                 weight_generation='naive', bias=1.0, output_bias=1.0,
                 output_input_scaling=1.0, input_density=1.0, solver='pinv', regression_parameters={}):

        super(ESN, self).__init__(n_input=n_input, n_reservoir=n_reservoir, n_output=n_output, spectral_radius=spectral_radius,
                                  noise_level=noise_level, input_scaling=input_scaling, leak_rate=leak_rate, sparseness=sparseness,
                                  random_seed=random_seed, out_activation=out_activation, out_inverse_activation=out_inverse_activation,
                                  weight_generation=weight_generation, bias=bias, output_bias=output_bias, output_input_scaling=output_input_scaling,
                                  input_density=input_density)


        self._solver = solver
        self._regression_parameters = regression_parameters

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
    def fit(self, inputData, outputData, transient_quota=0.05, verbose=0):
        #check the input data
        if inputData.shape[0] != outputData.shape[0]:
            raise ValueError("Amount of input and output datasets is not equal - {0} != {1}".format(inputData.shape[0], outputData.shape[0]))

        trainLength = inputData.shape[0]

        skipLength = int(trainLength*transient_quota)

        #define states' matrix
        X = np.zeros((1+self.n_input+self.n_reservoir,trainLength-skipLength))

        self._x = np.zeros((self.n_reservoir,1))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=trainLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(trainLength):
            u = super(ESN, self).update(inputData[t])
            if (t >= skipLength):
                #add valueset to the states' matrix
                X[:,t-skipLength] = np.vstack((self.output_bias, self.output_input_scaling*u, self._x))[:,0]
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        #define the target values
        Y_target = self.out_inverse_activation(outputData).T[:,skipLength:]

        if (self._solver == "pinv"):
            self._W_out = np.dot(Y_target, np.linalg.pinv(X))

            #calculate the training prediction now
            train_prediction = self.out_activation((np.dot(self._W_out, X)).T)

        elif (self._solver == "lsqr"):
            self._W_out = np.dot(np.dot(Y_target, X.T),np.linalg.inv(np.dot(X,X.T) + self._regression_parameters[0]*np.identity(1+self.n_input+self.n_reservoir)))

            """
                #alternative represantation of the equation

                Xt = X.T

                A = np.dot(X, Y_target.T)

                B = np.linalg.inv(np.dot(X, Xt)  + regression_parameter*np.identity(1+self.n_input+self.n_reservoir))

                self._W_out = np.dot(B, A)
                self._W_out = self._W_out.T
            """

            #calculate the training prediction now
            train_prediction = self.out_activation(np.dot(self._W_out, X).T)

        elif (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
            mode = self._solver[8:]
            params = self._regression_parameters
            params["solver"] = mode
            self._ridgeSolver = Ridge(**params)

            self._ridgeSolver.fit(X.T, Y_target.T)

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(X.T))

        elif (self._solver in ["sklearn_svr", "sklearn_svc"]):
            self._ridgeSolver = SVR(**self._regression_parameters)

            self._ridgeSolver.fit(X.T, Y_target.T.ravel())

            #calculate the training prediction now
            train_prediction = self.out_activation(self._ridgeSolver.predict(X.T))

        X = None

        #calculate the training error now
        training_error = np.sqrt(np.mean((train_prediction - outputData[skipLength:])**2))
        return training_error


    """
        Use the ESN in the generative mode to generate a signal autonomously.
    """
    def generate(self, n, initial_input, continuation=True, initial_data=None, update_processor=lambda x:x):
        #check the input data
        if (self.n_input != self.n_output):
            raise ValueError("n_input does not equal n_output. The generation mode uses its own output as its input. Therefore, n_input has to be equal to n_output - please adjust these numbers!")

        #let some input run through the ESN to initialize its states from a new starting value
        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    super(ESN, self).update(initial_data[t])

        predLength = n

        Y = np.zeros((self.n_output,predLength))
        inputData = initial_input
        for t in range(predLength):
            #update the inner states
            u = super(ESN, self).update(inputData)

            #calculate the prediction using the trained model
            if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd"]):
                y = self._ridgeSolver.predict(np.vstack((self.output_bias, self.output_input_scaling*u, self._x)).T)
            else:
                y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

            #apply the output activation function
            y = self.out_activation(y[:,0])
            Y[:,t] = update_processor(y)
            inputData = y

        #return the result
        return Y.T

    """
        Use the ESN in the predictive mode to predict the output signal by using an input signal.
    """
    def predict(self, inputData, continuation=True, initial_data=None, update_processor=lambda x:x, verbose=0):
        #let some input run through the ESN to initialize its states from a new starting value
        if (not continuation):
            self._x = np.zeros(self._x.shape)

            if (initial_data is not None):
                for t in range(initial_data.shape[0]):
                    super(ESN, self).update(initial_data[t])

        predLength = inputData.shape[0]

        Y = np.zeros((self.n_output,predLength))

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=predLength, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for t in range(predLength):
            #update the inner states
            u = super(ESN, self).update(inputData[t])

            #calculate the prediction using the trained model
            if (self._solver in ["sklearn_auto", "sklearn_lsqr", "sklearn_sag", "sklearn_svd", "sklearn_svr"]):
                y = self._ridgeSolver.predict(np.vstack((self.output_bias, self.output_input_scaling*u, self._x)).T).reshape((-1,1))
            else:
                y = np.dot(self._W_out, np.vstack((self.output_bias, self.output_input_scaling*u, self._x)))

            #apply the output activation function
            Y[:,t] = update_processor(self.out_activation(y[:,0]))
            if (verbose > 0):
                bar.update(t)

        if (verbose > 0):
            bar.finish()

        #return the result
        return Y.T
