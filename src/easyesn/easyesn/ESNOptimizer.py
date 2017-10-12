import numpy as np
from . import helper as hlp
from . import backend as B

class ESNOptimizer(object):

    def __init__(self):
        pass



    ####################################################################################################################################################

    # f`(X)
    def activationDerivation(self, X):
        return 4 / (2 + B.exp(2 * X) + B.exp(-2 * X))

    # del x / del alpha
    def derivationForLeakingRate(self, reservoir, oldDerivative, u, x):
        a = reservoir._leakingRate
        X = reservoir.calculateLinearNetworkTransmissions(u)
        return (1-a) * oldDerivative - x + reservoir._activation(X) + a * self.activationDerivation(X) * B.dot(reservoir._W, oldDerivative )

    # del x / del rho
    def derivationForSpectralRadius(self, reservoir, W_uniform, oldDerivative, u, x):
        a = reservoir._leakingRate
        X = reservoir.calculateLinearNetworkTransmissions(u)
        return (1-a) * oldDerivative + a * self.activationDerivation(X) * ( B.dot(reservoir._W, oldDerivative) + B.dot(W_uniform, x) )

    # del x / del s_in
    def derivationForInputScaling(self, reservoir, W_in_uniform, oldDerivative, u, x):
        a = reservoir._leakingRate
        X = reservoir.calculateLinearNetworkTransmissions(u)
        u = B.vstack((1,u))
        return (1-a)  * oldDerivative + a * self.activationDerivation(X) * ( B.dot(reservoir._W, oldDerivative) + B.dot(W_in_uniform, u) )

    # del W_out / del beta
    def derivationForPenalty(self, reservoir, Y, X, penalty):
        X_T = X.T
        term2 = B.inv( B.dot( X, X_T ) + penalty * B.identity(1 + reservoir.n_input + reservoir.n_reservoir) )
        return - B.dot( B.dot( Y, X_T ), B.dot( term2, term2 ) )

    # del W_out / del (alpha, rho or s_in)
    def derivationWoutForP(self, reservoir, Y, X, XPrime):
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

        term1 = B.inv(B.dot(X,X_T) + reservoir._regression_parameters[0]*B.identity(1 + reservoir.n_input + reservoir.n_reservoir))
        term2 = B.dot( XPrime, X_T) + B.dot( X, XPrime_T )

        return B.dot( Y, B.dot( XPrime_T, term1 ) - B.dot( B.dot( B.dot( X_T, term1 ), term2 ), term1 ) )


    ####################################################################################################################################################


    def optimizeParameterForTrainError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, learningRate=0.0001, epochs=1, transientTime = None):

        if not np.isscalar(reservoir._inputScaling):
            raise ValueError("Only penalty optimization is supported for a multiple input scalings at the moment. We are working on it.")

        # calculate stuff
        trainLength = trainInputs.shape[0]
        if transientTime is None:
            transientTime = reservoir.estimateTransientTime(trainInputs, trainTargets)

        # initializations of arrays:
        Ytarget = trainTargets[transientTime:].T
        #TODO handle different shapes with [None,:]...

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
        srGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))
        lrGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))
        isGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))

        # initialize fallback Parameter
        oldSR = reservoir._spectralRadius
        oldLR = reservoir._leakingRate
        oldIS = reservoir._inputScaling

        # initialize self.designMatrix and self.W_out
        oldLoss = reservoir.fit(trainInputs, trainTargets, transientTime=transientTime)

        # Calculate uniform matrices
        W_uniform = reservoir._W / reservoir._spectralRadius
        W_in_uniform = reservoir._W_input / reservoir._inputScaling

        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = B.zeros((reservoir.n_reservoir, 1))
            derivationLeakingRate = B.zeros((reservoir.n_reservoir, 1))
            derivationInputScaling = B.zeros((reservoir.n_reservoir, 1))

            # initialize the neuron states new
            x = B.zeros((reservoir.n_reservoir, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t, u in enumerate(trainInputs):
                u = u.reshape(-1, 1)
                oldx = x
                reservoir.update(u)
                x = reservoir._x
                # x = reservoir._X[2:,t]


                # calculate the del /x del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)
                if t >= transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    derivationConcatinationLeakingRate = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    derivationConcatinationInputScaling = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                    # add to matrix
                    srGradientsMatrix[:, t - transientTime] = derivationConcatinationSpectralRadius
                    lrGradientsMatrix[:, t - transientTime] = derivationConcatinationLeakingRate
                    isGradientsMatrix[:, t - transientTime] = derivationConcatinationInputScaling

            # calculate del W_out / del (rho, alpha, s_in) based on the designMatrix and the derivative of the designMatrix we just calculated
            WoutPrimeSR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, srGradientsMatrix)
            WoutPrimeLR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, lrGradientsMatrix)
            WoutPrimeIS = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, isGradientsMatrix)

            # reinitialize the states
            x = B.zeros((reservoir.n_reservoir, 1))

            # go through the train time again, and this time, calculate del error / del (rho, alpha, s_in) based on del W_out and the single derivatives
            for t, u in enumerate(trainInputs):
                u = u.reshape(-1, 1)
                reservoir.update(u)
                x = reservoir._x
                if t >= transientTime:

                    # calculate error at given time step
                    error = (trainTargets[t] - B.dot( reservoir._W_out, B.vstack((1, u, x)) ) ).T

                    # calculate gradients
                    gradientSR = B.dot(-error, B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir._W_out, srGradientsMatrix[:, t - transientTime]))
                    srGradients[t - transientTime] = gradientSR
                    gradientLR = B.dot(-error, B.dot(WoutPrimeLR, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir._W_out, lrGradientsMatrix[:, t - transientTime]))
                    lrGradients[t - transientTime] = gradientLR
                    gradientIS = B.dot(-error, B.dot(WoutPrimeIS, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir._W_out, isGradientsMatrix[:, t - transientTime]))
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
            reservoir.setSpectralRadius(reservoir._spectralRadius - learningRate * gradientSR)

            # update leaking rate
            reservoir.setLeakingRate( reservoir._leakingRate - learningRate * gradientLR )

            # update input scaling
            reservoir.setInputScaling(reservoir._inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = reservoir.fit(trainInputs, trainTargets, transientTime=transientTime)
            validationLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

            if fitLoss > oldLoss:
                reservoir.setSpectralRadius(oldSR)
                reservoir.setLeakingRate(oldLR)
                reservoir.setInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = reservoir._spectralRadius
                oldLR = reservoir._leakingRate
                oldIS = reservoir._inputScaling
                oldLoss = fitLoss
                spectralRadiuses.append(reservoir._spectralRadius)
                leakingRates.append(reservoir._leakingRate)
                inputScalings.append(reservoir._inputScaling)
                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)


        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)


    def optimizeParameterForEvaluationError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, optimizationLength, learningRate=0.0001, epochs=1, transientTime = None):

        if not np.isscalar(reservoir._inputScaling):
            raise ValueError("Only penalty optimization is supported for a multiple input scalings at the moment. We are working on it.")

        # calculate stuff
        trainLength = trainInputs.shape[0]
        if transientTime is None:
            transientTime = reservoir.estimateTransientTime(trainInputs, trainTargets)

        # initializations of arrays:
        Ytarget = trainTargets[transientTime:].T
        # TODO handle different shapes with [None,:]...

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
        srGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))
        lrGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))
        isGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - transientTime))

        # initialize variables for the "when the error goes up, go back and divide learning rate by 2" mechanism
        oldSR = reservoir._spectralRadius
        oldLR = reservoir._leakingRate
        oldIS = reservoir._inputScaling

        # initialize self.designMatrix and self.W_out
        reservoir.fit(trainInputs, trainTargets, transientTime=transientTime)
        oldLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

        # Calculate uniform matrices
        W_uniform = reservoir._W / reservoir._spectralRadius
        W_in_uniform = reservoir._W_input / reservoir._inputScaling

        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = B.zeros((reservoir.n_reservoir, 1))
            derivationLeakingRate = B.zeros((reservoir.n_reservoir, 1))
            derivationInputScaling = B.zeros((reservoir.n_reservoir, 1))
            x = B.zeros((reservoir.n_reservoir, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t, u in enumerate(trainInputs):
                u = u.reshape(-1, 1)
                oldx = x
                x = reservoir._X[2:,t]


                # calculate the del x/ del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)
                if t >= transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    srGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    lrGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    isGradientsMatrix[:, t - transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

            # add to matrix
            WoutPrimeSR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, srGradientsMatrix)
            WoutPrimeLR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, lrGradientsMatrix)
            WoutPrimeIS = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, isGradientsMatrix)

            # this time go through validation length
            for t, u in enumerate(validationInputs):
                u = u.reshape(-1, 1)
                oldx = x
                reservoir.update(u)
                x = reservoir._x

                # calculate error at given time step
                error = (validationTargets[t] - B.dot( reservoir._W_out, B.vstack((1, u, x)) ) ).T

                # calculate del x / del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)

                # concatenate derivations with 0
                derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                derivationConcatinationLeakingRate = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                derivationConcatinationInputScaling = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                # calculate gradients
                gradientSR = B.dot(-error, B.dot(reservoir._W_out, derivationConcatinationSpectralRadius) + B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]))
                srGradients[t] = gradientSR
                gradientLR = B.dot(-error, B.dot(reservoir._W_out, derivationConcatinationLeakingRate) + B.dot(WoutPrimeLR, B.vstack( (1, u, x))[:, 0]))
                lrGradients[t] = gradientLR
                gradientIS = B.dot(-error, B.dot(reservoir._W_out, derivationConcatinationInputScaling) + B.dot(WoutPrimeIS, B.vstack( (1, u, x))[:, 0]))
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
            reservoir.setSpectralRadius(reservoir._spectralRadius - learningRate * gradientSR)

            # update leaking rate
            reservoir.setLeakingRate(reservoir._leakingRate - learningRate * gradientLR)

            # update input scaling
            reservoir.setInputScaling(reservoir._inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = reservoir.fit(trainInputs, trainTargets, transientTime=transientTime)
            validationLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if validationLoss > oldLoss:
                reservoir.setSpectralRadius(oldSR)
                reservoir.setLeakingRate(oldLR)
                reservoir.setInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = reservoir._spectralRadius
                oldLR = reservoir._leakingRate
                oldIS = reservoir._inputScaling
                oldLoss = validationLoss

                spectralRadiuses.append(reservoir._spectralRadius)
                leakingRates.append(reservoir._leakingRate)
                inputScalings.append(reservoir._inputScaling)

                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)



        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)


    def optimizePenaltyForEvaluationError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, optimizationLength, learningRate=0.0001, epochs=1, penalty=0.1, transientTime=0):

        Ytarget = trainTargets[transientTime].T
        # TODO handle different shapes with [None,:]...
        penalty = penalty

        fitLosses = list()
        validationLosses = list()
        penalties = list()

        penaltyDerivatives = B.zeros(optimizationLength)
        oldPenalty = penalty

        reservoir.fit(trainInputs, trainTargets, transientTime=transientTime)
        oldLoss = hlp.loss(reservoir.predict(validationInputs), validationTargets)

        evaluationEchoFunction = B.zeros((1 + reservoir.n_reservoir + reservoir.n_input, optimizationLength))
        x = reservoir.x

        for t, u in enumerate(validationInputs):
            u = u.reshape(-1, 1)
            reservoir.update(u)
            x = reservoir._x
            evaluationEchoFunction[:, t] = B.vstack((1, u, x)).squeeze()


        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            penaltyDerivative = self.derivationForPenalty(reservoir, Ytarget, reservoir._X, penalty)

            for t in range(len(validationInputs)):
                predictionPoint = B.dot(reservoir._W_out, evaluationEchoFunction[:, t].reshape(-1, 1))
                error = (trainTargets[t] - predictionPoint).T
                penaltyDerivatives[t] = - B.dot(B.dot(error, penaltyDerivative), predictionPoint)

            penaltyGradient = sum(penaltyDerivatives)

            penaltyGradient = B.sign( penaltyGradient )

            penalty = penalty - learningRate * penaltyGradient

            reservoir.setPenalty(penalty)
            reservoir._calculateOutputMatrix()
            fitLoss = hlp.loss( reservoir.predict( trainInputs), trainTargets )
            validationLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

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

        return (validationLosses, fitLosses, penalties)


