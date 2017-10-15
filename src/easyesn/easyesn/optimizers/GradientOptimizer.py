import numpy as np
from .. import helper as hlp
from .. import backend as B

import progressbar

class GradientOptimizer(object):
    def _validateReservoir(self):
        if not np.isclose(self._reservoir.noise_level, 0.0):
            raise ValueError("Noise must be set to 0.0 for this optimizer, but it is set to {0}.".format(self._reservoir.noise_level))
        if self._reservoir._solver is not "lsqr":
            raise ValueError("The reservoir's solver must be set to 'lsqr' (Ridge Regression) for this optimizer.")

    def __init__(self, reservoir, learningRate=0.0001):
        self._reservoir = reservoir
        self._validateReservoir()
        self.setLearningRate(learningRate)


    def setLearningRate(self, learningRate):
        if np.isscalar(learningRate):
            self.learningRates = (learningRate, learningRate, learningRate)
        else:
            if len(learningRate) != 3:
                raise ValueError("LearningRate has to be a scalar or a list/tuple with 3 entries.")
            else:
                self.learningRates = learningRate

    ####################################################################################################################################################

    # f`(X)
    def activationDerivation(self, X):
        return 4 / (2 + B.exp(2 * X) + B.exp(-2 * X))

    def _derivationLrSrIs(self, oldDerivativeLr, oldDerivativeSr, oldDerivativeIs, W_uniform, W_in_uniform, u, x):
        a = self._reservoir._leakingRate
        X = self._reservoir.calculateLinearNetworkTransmissions(u)

        activationX = self._reservoir._activation(X)
        activationDerivationX = self.activationDerivation(X)

        dLr = (1-a) * oldDerivativeLr - x + activationX + a * activationDerivationX * B.dot(self._reservoir._W, oldDerivativeLr)
        dSr = (1-a) * oldDerivativeSr + a * activationDerivationX * (B.dot(self._reservoir._W, oldDerivativeSr) + B.dot(W_uniform, x))

        u = B.vstack((1,u))
        dIs = (1-a) * oldDerivativeIs + a * activationDerivationX * (B.dot(self._reservoir._W, oldDerivativeIs) + B.dot(W_in_uniform, u))

        return dLr, dSr, dIs

    # del W_out / del beta
    def derivationForPenalty(self, Y, X, penalty):
        X_T = X.T
        term2 = B.inv( B.dot( X, X_T ) + penalty * B.identity(1 + self._reservoir.n_input + self._reservoir.n_self._reservoir) )
        return - B.dot( B.dot( Y, X_T ), B.dot( term2, term2 ) )

    # del W_out / del (alpha, rho or s_in)
    def derivationWoutForP(self, Y, X, XPrime):
        X_T = X.T
        XPrime_T = XPrime.T

        # A = dot(X,X_T) + penalty*eye(1 + self.target_dim + self.n_reservoir)
        # APrime = dot( XPrime, X_T) + dot( X, XPrime_T )
        # APrime_T = APrime.T
        # InvA = B.inv(A)
        # InvA_T = InvA.T
        #
        # term1 = dot(XPrime_T, InvA)
        #
        # term21 = -dot( InvA, dot( APrime, InvA ) )
        # term22 = dot( dot( dot( InvA, InvA_T), APrime_T), eye(1 + self.target_dim + self.n_reservoir) - dot( A, InvA ) )
        # term23 = dot( dot( eye(1 + self.target_dim + self.n_reservoir) - dot( InvA, A ), APrime_T), dot( InvA_T, InvA) )
        # term2 = dot( X_T, term21 + term22 + term23 )
        #
        # return dot( Y, term1 + term2)

        term1 = B.inv(B.dot(X,X_T) + self._reservoir._regression_parameters[0]*B.identity(1 + self._reservoir.n_input + self._reservoir.n_reservoir))
        term2 = B.dot( XPrime, X_T) + B.dot( X, XPrime_T )

        return B.dot( Y, B.dot( XPrime_T, term1 ) - B.dot( B.dot( B.dot( X_T, term1 ), term2 ), term1 ) )
    ####################################################################################################################################################


    def fit(self, trainingInput, trainingOutput, validationInput, validationOutput, verbose=1):
        self.optimizeParameterForTrainError(trainingInput, trainingOutput, validationInput, validationOutput, verbose)
        self.optimizePenaltyForEvaluationError(trainingInput, trainingOutput, validationInput, validationOutput, verbose)
        self.optimizeParameterForValidationError(trainingInput, trainingOutput, validationInput, validationOutput, verbose)

    def optimizeParameterForTrainError(self, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                                       epochs=1, transientTime=None, verbose=1):

        self._validateReservoir()

        learningRate = self.learningRates[0]

        if not np.isscalar(self._reservoir._inputScaling):
            raise ValueError("Only penalty optimization is supported for a multiple input scalings at the moment. We are working on it.")

        # calculate stuff
        trainLength = trainingInputData.shape[0]
        if transientTime is None:
            transientTime = self._reservoir.estimateTransientTime(trainingInputData, trainingOutputData)

        # initializations of arrays:
        if (len(trainingOutputData.shape) == 1):
            Ytarget = trainingOutputData[None, transientTime:].T
        else:
            Ytarget = trainingOutputData[transientTime:].T

        if (len(validationOutputData.shape) == 1):
            validationOutputData = validationOutputData[:, None]

        Ytarget = self._reservoir.out_inverse_activation(Ytarget)
        validationOutputData = self._reservoir.out_inverse_activation(validationOutputData)

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        fitLosses = list()
        validationLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = B.zeros(trainLength - transientTime)
        lrGradients = B.zeros(trainLength - transientTime)
        isGradients = B.zeros(trainLength - transientTime)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))
        lrGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))
        isGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))

        # initialize fallback Parameter
        oldSR = self._reservoir._spectralRadius
        oldLR = self._reservoir._leakingRate
        oldIS = self._reservoir._inputScaling

        # initialize self.designMatrix and self.W_out
        oldLoss = self._reservoir.fit(trainingInputData, trainingOutputData, transientTime=transientTime)

        # Calculate uniform matrices
        W_uniform = self._reservoir._W / self._reservoir._spectralRadius
        W_in_uniform = self._reservoir._W_input / self._reservoir._inputScaling

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=epochs, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for epoch in range(epochs):
            if verbose > 0:
                bar.update(epoch)

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = B.zeros((self._reservoir.n_reservoir, 1))
            derivationLeakingRate = B.zeros((self._reservoir.n_reservoir, 1))
            derivationInputScaling = B.zeros((self._reservoir.n_reservoir, 1))

            # initialize the neuron states new
            x = B.zeros((self._reservoir.n_reservoir, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t, u in enumerate(trainingInputData):
                u = u.reshape(-1, 1)
                oldx = x
                self._reservoir.update(u)
                x = self._reservoir._x
                # x = self._reservoir._X[2:,t]


                # calculate the del /x del (rho, alpha, s_in)
                #derivationSpectralRadius = self.derivationForSpectralRadius(W_uniform, derivationSpectralRadius, u, oldx)
                #derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                #derivationInputScaling = self.derivationForInputScaling(W_in_uniform, derivationInputScaling, u, oldx)

                derivationLeakingRate, derivationSpectralRadius, derivationInputScaling = self._derivationLrSrIs(derivationLeakingRate, derivationSpectralRadius, derivationInputScaling, W_uniform, W_in_uniform, u, oldx)
                if t >= transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    derivationConcatinationLeakingRate = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    derivationConcatinationInputScaling = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                    # add to matrix
                    srGradientsMatrix[:, t - transientTime] = derivationConcatinationSpectralRadius
                    lrGradientsMatrix[:, t - transientTime] = derivationConcatinationLeakingRate
                    isGradientsMatrix[:, t - transientTime] = derivationConcatinationInputScaling

            # calculate del W_out / del (rho, alpha, s_in) based on the designMatrix and the derivative of the designMatrix we just calculated
            WoutPrimeSR = self.derivationWoutForP(Ytarget, self._reservoir._X, srGradientsMatrix)
            WoutPrimeLR = self.derivationWoutForP(Ytarget, self._reservoir._X, lrGradientsMatrix)
            WoutPrimeIS = self.derivationWoutForP(Ytarget, self._reservoir._X, isGradientsMatrix)

            # reinitialize the states
            x = B.zeros((self._reservoir.n_reservoir, 1))

            # go through the train time again, and this time, calculate del error / del (rho, alpha, s_in) based on del W_out and the single derivatives
            for t, u in enumerate(trainingInputData):
                u = u.reshape(-1, 1)
                self._reservoir.update(u)
                x = self._reservoir._x
                if t >= transientTime:

                    # calculate error at given time step
                    error = (trainingOutputData[t] - B.dot( self._reservoir._W_out, B.vstack((1, u, x)) ) ).T

                    # calculate gradients
                    gradientSR = B.dot(-error, B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]) + B.dot(self._reservoir._W_out, srGradientsMatrix[:, t - transientTime]))
                    srGradients[t - transientTime] = gradientSR
                    gradientLR = B.dot(-error, B.dot(WoutPrimeLR, B.vstack((1, u, x))[:, 0]) + B.dot(self._reservoir._W_out, lrGradientsMatrix[:, t - transientTime]))
                    lrGradients[t - transientTime] = gradientLR
                    gradientIS = B.dot(-error, B.dot(WoutPrimeIS, B.vstack((1, u, x))[:, 0]) + B.dot(self._reservoir._W_out, isGradientsMatrix[:, t - transientTime]))
                    isGradients[t - transientTime] = gradientIS

            # sum up the gradients del error / del (rho, alpha, s_in) to get final gradient
            gradientSR = sum(srGradients)
            gradientLR = sum(lrGradients)
            gradientIS = sum(isGradients)

            # normalize gradients to length 1
            gradientVectorLength = B.sqrt(gradientSR ** 2 + gradientLR ** 2 + gradientIS ** 2)

            gradientSR /= gradientVectorLength
            gradientLR /= gradientVectorLength
            gradientIS /= gradientVectorLength

            # update spectral radius
            self._reservoir.setSpectralRadius(self._reservoir._spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self._reservoir.setLeakingRate( self._reservoir._leakingRate - learningRate * gradientLR )

            # update input scaling
            self._reservoir.setInputScaling(self._reservoir._inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = self._reservoir.fit(trainingInputData, trainingOutputData, transientTime=transientTime)
            validationLoss = hlp.loss( self._reservoir.predict(validationInputData), validationOutputData )

            if fitLoss > oldLoss:
                self._reservoir.setSpectralRadius(oldSR)
                self._reservoir.setLeakingRate(oldLR)
                self._reservoir.setInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = self._reservoir._spectralRadius
                oldLR = self._reservoir._leakingRate
                oldIS = self._reservoir._inputScaling
                oldLoss = fitLoss
                spectralRadiuses.append(self._reservoir._spectralRadius)
                leakingRates.append(self._reservoir._leakingRate)
                inputScalings.append(self._reservoir._inputScaling)
                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)

        if verbose > 0:
            bar.finish()

        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)

    def optimizeParameterForValidationError(self, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                                            optimizationLength, epochs=1, transientTime=None, verbose=1):

        self._validateReservoir()

        learningRate = self.learningRates[1]

        if not np.isscalar(self._reservoir._inputScaling):
            raise ValueError("Only penalty optimization is supported for a multiple input scalings at the moment. We are working on it.")

        # calculate stuff
        trainLength = trainingInputData.shape[0]
        if transientTime is None:
            transientTime = self._reservoir.estimateTransientTime(trainingInputData, trainingOutputData)

        # initializations of arrays:
        if (len(trainingOutputData.shape) == 1):
            Ytarget = trainingOutputData[None, transientTime:].T
        else:
            Ytarget = trainingOutputData[transientTime:].T

        if (len(validationOutputData.shape) == 1):
            validationOutputData = validationOutputData[:, None]

        Ytarget = self._reservoir.out_inverse_activation(Ytarget)
        validationOutputData = self._reservoir.out_inverse_activation(validationOutputData)

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        fitLosses = list()
        validationLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = B.zeros(optimizationLength)
        lrGradients = B.zeros(optimizationLength)
        isGradients = B.zeros(optimizationLength)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))
        lrGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))
        isGradientsMatrix = B.zeros((self._reservoir.n_reservoir + self._reservoir.n_input + 1, trainLength - transientTime))

        # initialize variables for the "when the error goes up, go back and divide learning rate by 2" mechanism
        oldSR = self._reservoir._spectralRadius
        oldLR = self._reservoir._leakingRate
        oldIS = self._reservoir._inputScaling

        # initialize self.designMatrix and self.W_out
        self._reservoir.fit(trainingInputData, trainingOutputData, transientTime=transientTime)
        oldLoss = hlp.loss( self._reservoir.predict(validationInputData), validationOutputData )

        # Calculate uniform matrices
        W_uniform = self._reservoir._W / self._reservoir._spectralRadius
        W_in_uniform = self._reservoir._W_input / self._reservoir._inputScaling

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=epochs, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for epoch in range(epochs):
            if verbose > 0:
                bar.update(epoch)

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = B.zeros((self._reservoir.n_reservoir, 1))
            derivationLeakingRate = B.zeros((self._reservoir.n_reservoir, 1))
            derivationInputScaling = B.zeros((self._reservoir.n_reservoir, 1))
            x = B.zeros((self._reservoir.n_reservoir, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t, u in enumerate(trainingInputData):
                u = u.reshape(-1, 1)
                oldx = x
                x = self._reservoir._X[2:,t]


                # calculate the del /x del (rho, alpha, s_in)
                #derivationSpectralRadius = self.derivationForSpectralRadius(W_uniform, derivationSpectralRadius, u, oldx)
                #derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                #derivationInputScaling = self.derivationForInputScaling(W_in_uniform, derivationInputScaling, u, oldx)

                derivationLeakingRate, derivationSpectralRadius, derivationInputScaling = self._derivationLrSrIs(derivationLeakingRate, derivationSpectralRadius, derivationInputScaling, W_uniform, W_in_uniform, u, oldx)
                if t >= transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    srGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    lrGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    isGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

            # add to matrix
            WoutPrimeSR = self.derivationWoutForP(Ytarget, self._reservoir._X, srGradientsMatrix)
            WoutPrimeLR = self.derivationWoutForP(Ytarget, self._reservoir._X, lrGradientsMatrix)
            WoutPrimeIS = self.derivationWoutForP(Ytarget, self._reservoir._X, isGradientsMatrix)

            # this time go through validation length
            for t, u in enumerate(validationInputData):
                u = u.reshape(-1, 1)
                oldx = x
                self._reservoir.update(u)
                x = self._reservoir._x

                # calculate error at given time step
                error = (validationOutputData[t] - B.dot( self._reservoir._W_out, B.vstack((1, u, x)) ) ).T

# calculate the del /x del (rho, alpha, s_in)
                #derivationSpectralRadius = self.derivationForSpectralRadius(W_uniform, derivationSpectralRadius, u, oldx)
                #derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                #derivationInputScaling = self.derivationForInputScaling(W_in_uniform, derivationInputScaling, u, oldx)

                derivationLeakingRate, derivationSpectralRadius, derivationInputScaling = self._derivationLrSrIs(derivationLeakingRate, derivationSpectralRadius, derivationInputScaling, W_uniform, W_in_uniform, u, oldx)

                # concatenate derivations with 0
                derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                derivationConcatinationLeakingRate = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                derivationConcatinationInputScaling = B.concatenate( (B.zeros(self._reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                # calculate gradients
                gradientSR = B.dot(-error, B.dot(self._reservoir._W_out, derivationConcatinationSpectralRadius) + B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]))
                srGradients[t] = gradientSR
                gradientLR = B.dot(-error, B.dot(self._reservoir._W_out, derivationConcatinationLeakingRate) + B.dot(WoutPrimeLR, B.vstack( (1, u, x))[:, 0]))
                lrGradients[t] = gradientLR
                gradientIS = B.dot(-error, B.dot(self._reservoir._W_out, derivationConcatinationInputScaling) + B.dot(WoutPrimeIS, B.vstack( (1, u, x))[:, 0]))
                isGradients[t] = gradientIS

            # sum up the gradients del error / del (rho, alpha, s_in) to get final gradient
            gradientSR = sum(srGradients)
            gradientLR = sum(lrGradients)
            gradientIS = sum(isGradients)

            # normalize length of gradient to 1
            gradientVectorLength = B.sqrt(gradientSR ** 2 + gradientLR ** 2 + gradientIS ** 2)

            gradientSR /= gradientVectorLength
            gradientLR /= gradientVectorLength
            gradientIS /= gradientVectorLength

            # update spectral radius
            self._reservoir.setSpectralRadius(self._reservoir._spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self._reservoir.setLeakingRate(self._reservoir._leakingRate - learningRate * gradientLR)

            # update input scaling
            self._reservoir.setInputScaling(self._reservoir._inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = self._reservoir.fit(trainingInputData, trainingOutputData, transientTime=transientTime)
            validationLoss = hlp.loss( self._reservoir.predict(validationInputData), validationOutputData )

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if validationLoss > oldLoss:
                self._reservoir.setSpectralRadius(oldSR)
                self._reservoir.setLeakingRate(oldLR)
                self._reservoir.setInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = self._reservoir._spectralRadius
                oldLR = self._reservoir._leakingRate
                oldIS = self._reservoir._inputScaling
                oldLoss = validationLoss

                spectralRadiuses.append(self._reservoir._spectralRadius)
                leakingRates.append(self._reservoir._leakingRate)
                inputScalings.append(self._reservoir._inputScaling)

                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)

        if verbose > 0:
            bar.finish()

        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)

    def optimizePenaltyForEvaluationError(self, trainingInputData, trainingOutputData, validationInputData, validationOutputData,
                                          optimizationLength, epochs=1, penalty=0.1, transientTime=0, verbose=1):

        self._validateReservoir()

        learningRate = self.learningRates[2]

        # initializations of arrays:
        if (len(trainingOutputData.shape) == 1):
            Ytarget = trainingOutputData[None, transientTime:].T
        else:
            Ytarget = trainingOutputData[transientTime:].T

        if (len(validationOutputData.shape) == 1):
            validationOutputData = validationOutputData[:, None]

        Ytarget = self._reservoir.out_inverse_activation(Ytarget)
        validationOutputData = self._reservoir.out_inverse_activation(validationOutputData)

        fitLosses = list()
        validationLosses = list()
        penalties = list()

        penaltyDerivatives = B.zeros(optimizationLength)
        oldPenalty = penalty

        self._reservoir.fit(trainingInputData, trainingOutputData, transientTime=transientTime)
        oldLoss = hlp.loss(self._reservoir.predict(validationInputData), validationOutputData)

        evaluationEchoFunction = B.zeros((1 + self._reservoir.n_reservoir + self._reservoir.n_input, optimizationLength))
        x = self._reservoir.x

        for t, u in enumerate(validationInputData):
            u = u.reshape(-1, 1)
            self._reservoir.update(u)
            x = self._reservoir._x
            evaluationEchoFunction[:, t] = B.vstack((1, u, x)).squeeze()

        if (verbose > 0):
            bar = progressbar.ProgressBar(max_value=epochs, redirect_stdout=True, poll_interval=0.0001)
            bar.update(0)

        for epoch in range(epochs):
            if verbose > 0:
                bar.update(epoch)

            penaltyDerivative = self.derivationForPenalty(Ytarget, self._reservoir._X, penalty)

            for t in range(len(validationInputData)):
                predictionPoint = B.dot(self._reservoir._W_out, evaluationEchoFunction[:, t].reshape(-1, 1))
                error = (trainingOutputData[t] - predictionPoint).T
                penaltyDerivatives[t] = - B.dot(B.dot(error, penaltyDerivative), predictionPoint)

            penaltyGradient = sum(penaltyDerivatives)

            penaltyGradient = B.sign( penaltyGradient )

            penalty = penalty - learningRate * penaltyGradient

            self._reservoir.setPenalty(penalty)
            self._reservoir._calculateOutputMatrix()
            fitLoss = hlp.loss( self._reservoir.predict( trainingInputData), trainingOutputData )
            validationLoss = hlp.loss( self._reservoir.predict(validationInputData), validationOutputData )

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if validationLoss > oldLoss:
                penalty = oldPenalty
                learningRate = learningRate / 2
            else:
                oldPenalty = penalty
                oldLoss = validationLoss
                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)
                penalties.append(penalty)

        if verbose > 0:
            bar.finish()

        return (validationLosses, fitLosses, penalties)
