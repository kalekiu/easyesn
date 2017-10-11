#from numpy import *
from . import helper as hlp
from . import backend as B

class Optimizer:

    def __init__(self, backend):
        B = backend



    ####################################################################################################################################################

    # f`(X)
    def activationDerivation(self, X):
        return 4 / (2 + B.exp(2 * X) + B.exp(-2 * X))

    # del x / del alpha
    def derivationForLeakingRate(self, reservoir, oldDerivative, u, x):
        a = reservoir.leakingRate
        X = reservoir.X(x, u)
        return (1-a) * oldDerivative - x + reservoir.f(X) + a * self.activationDerivation(X) * B.dot(reservoir.W, oldDerivative )

    # del x / del rho
    def derivationForSpectralRadius(self, reservoir, W_uniform, oldDerivative, u, x):
        a = reservoir.leakingRate
        X = reservoir.X(x, u)
        return (1-a) * oldDerivative + a * self.activationDerivation(X) * ( B.dot(reservoir.W, oldDerivative) + B.dot(W_uniform, x) )

    # del x / del s_in
    def derivationForInputScaling(self, reservoir, W_in_uniform, oldDerivative, u, x):
        a = reservoir.leakingRate
        X = reservoir.X(x, u)
        u = B.vstack((1,u))
        return (1-a)  * oldDerivative + a * self.activationDerivation(X) * ( B.dot(reservoir.W, oldDerivative) + B.dot(W_in_uniform, u) )

    # del W_out / del beta
    def derivationForPenalty(self, reservoir, Y, X, penalty):
        X_T = X.T
        term2 = B.linalg.inv( B.dot( X, X_T ) + penalty * B.eye(1 + reservoir.n_input + reservoir.n_reservoir) )
        return - B.dot( B.dot( Y, X_T ), B.dot( term2, term2 ) )

    # del W_out / del (alpha, rho or s_in)
    def derivationWoutForP(self, reservoir, Y, X, XPrime, penalty):
        X_T = X.T
        XPrime_T = XPrime.T

        # A = dot(X,X_T) + penalty*eye(1 + self.target_dim + self.n_reservoir)
        # APrime = dot( XPrime, X_T) + dot( X, XPrime_T )
        # APrime_T = APrime.T
        # InvA = linalg.inv(A)
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

        term1 = B.linalg.inv(B.dot(X,X_T) + penalty*B.eye(1 + reservoir.n_input + reservoir.n_reservoir))
        term2 = B.dot( XPrime, X_T) + B.dot( X, XPrime_T )

        return B.dot( Y, B.dot( XPrime_T, term1 ) - B.dot( B.dot( B.dot( X_T, term1 ), term2 ), term1 ) )


    ####################################################################################################################################################


    def optimizeParameterForTrainError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, trainLength, learningRate=0.0001, epochs=1, penalty=0.1, errorEvaluationLength=500):

        # initializations of arrays:
        Ytarget = trainTargets[None, :]

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        fitLosses = list()
        validationLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = B.zeros(trainLength - reservoir.transientTime)
        lrGradients = B.zeros(trainLength - reservoir.transientTime)
        isGradients = B.zeros(trainLength - reservoir.transientTime)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))
        lrGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))
        isGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))

        # initialize fallback Parameter
        oldSR = reservoir.spectralRadius
        oldLR = self.leakingRate
        oldIS = reservoir.inputScaling

        # initialize self.designMatrix and self.W_out
        oldLoss = reservoir.fit(trainInputs, trainTargets)

        # Calculate uniform matrices
        W_uniform = reservoir.W / reservoir.spectralRadius
        W_in_uniform = reservoir.W_in / reservoir.inputScaling

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
                x = reservoir._X[t]

                # calculate the del /x del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)
                if t >= reservoir.transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    derivationConcatinationLeakingRate = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    derivationConcatinationInputScaling = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                    # add to matrix
                    srGradientsMatrix[:, t - reservoir.transientTime] = derivationConcatinationSpectralRadius
                    lrGradientsMatrix[:, t - reservoir.transientTime] = derivationConcatinationLeakingRate
                    isGradientsMatrix[:, t - reservoir.transientTime] = derivationConcatinationInputScaling

            # calculate del W_out / del (rho, alpha, s_in) based on the designMatrix and the derivative of the designMatrix we just calculated
            WoutPrimeSR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, srGradientsMatrix, penalty)
            WoutPrimeLR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, lrGradientsMatrix, penalty)
            WoutPrimeIS = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, isGradientsMatrix, penalty)

            # reinitialize the states
            x = B.zeros((reservoir.n_reservoir, 1))

            # go through the train time again, and this time, calculate del error / del (rho, alpha, s_in) based on del W_out and the single derivatives
            for t, u in enumerate(trainInputs):
                u = u.reshape(-1, 1)
                reservoir.update(u)
                x = reservoir._x
                if t >= reservoir.transientTime:

                    # calculate error at given time step
                    error = (trainTargets[t] - B.dot( reservoir.W_out, B.vstack((1, u, x)) ) ).T

                    # calculate gradients
                    gradientSR = B.dot(-error, B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir.W_out, srGradientsMatrix[:, t - reservoir.transientTime]))
                    srGradients[t - reservoir.transientTime] = gradientSR
                    gradientLR = B.dot(-error, B.dot(WoutPrimeLR, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir.W_out, lrGradientsMatrix[:, t - reservoir.transientTime]))
                    lrGradients[t - reservoir.transientTime] = gradientLR
                    gradientIS = B.dot(-error, B.dot(WoutPrimeIS, B.vstack((1, u, x))[:, 0]) + B.dot(reservoir.W_out, isGradientsMatrix[:, t - reservoir.transientTime]))
                    isGradients[t - reservoir.transientTime] = gradientIS

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
            reservoir.tuneSpectralRadius(reservoir.spectralRadius - learningRate * gradientSR)

            # update leaking rate
            reservoir.leakingRate = reservoir.leakingRate - learningRate * gradientLR

            # update input scaling
            reservoir.tuneInputScaling(reservoir.inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = reservoir.fit(trainInputs, trainTargets)
            validationLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

            if fitLoss > oldLoss:
                reservoir.tuneSpectralRadius(oldSR)
                reservoir.leakingRate = oldLR
                reservoir.tuneInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = reservoir.spectralRadius
                oldLR = reservoir.leakingRate
                oldIS = reservoir.inputScaling
                oldLoss = fitLoss
                spectralRadiuses.append(reservoir.spectralRadius)
                leakingRates.append(reservoir.leakingRate)
                inputScalings.append(reservoir.inputScaling)
                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)


        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)


    def optimizeParameterForEvaluationError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, trainLength, optimizationLength, learningRate=0.0001, epochs=1, penalty=0.1):

        # initializations of arrays:
        Ytarget = trainTargets[None,:]

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
        srGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))
        lrGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))
        isGradientsMatrix = B.zeros((reservoir.n_reservoir + reservoir.n_input + 1, trainLength - reservoir.transientTime))

        # initialize variables for the "when the error goes up, go back and divide learning rate by 2" mechanism
        oldSR = reservoir.spectralRadius
        oldLR = reservoir.leakingRate
        oldIS = reservoir.inputScaling

        # initialize self.designMatrix and self.W_out
        reservoir.fit(trainInputs, trainTargets)
        oldLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

        # Calculate uniform matrices
        W_uniform = reservoir.W / reservoir.spectralRadius
        W_in_uniform = reservoir.W_in / reservoir.inputScaling

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
                x = reservoir._X[t]

                # calculate the del x/ del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)
                if t >= reservoir.transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    srGradientsMatrix[:, t - reservoir.transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                    lrGradientsMatrix[:, t - reservoir.transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                    isGradientsMatrix[:, t - reservoir.transientTime] = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

            # add to matrix
            WoutPrimeSR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, srGradientsMatrix, penalty)
            WoutPrimeLR = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, lrGradientsMatrix, penalty)
            WoutPrimeIS = self.derivationWoutForP(reservoir, Ytarget, reservoir._X, isGradientsMatrix, penalty)

            # this time go through validation length
            for t, u in enumerate(validationInputs):
                u = u.reshape(-1, 1)
                oldx = x
                reservoir.update(u)
                x = reservoir._x

                # calculate error at given time step
                error = (validationTargets[t] - B.dot( reservoir.W_out, B.vstack((1, u, x)) ) ).T

                # calculate del x / del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(reservoir, W_uniform, derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(reservoir, derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(reservoir, W_in_uniform, derivationInputScaling, u, oldx)

                # concatenate derivations with 0
                derivationConcatinationSpectralRadius = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationSpectralRadius[:, 0]), axis=0)
                derivationConcatinationLeakingRate = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationLeakingRate[:, 0]), axis=0)
                derivationConcatinationInputScaling = B.concatenate( (B.zeros(reservoir.n_input + 1), derivationInputScaling[:, 0]), axis=0)

                # calculate gradients
                gradientSR = B.dot(-error, B.dot(reservoir.W_out, derivationConcatinationSpectralRadius) + B.dot(WoutPrimeSR, B.vstack((1, u, x))[:, 0]))
                srGradients[t] = gradientSR
                gradientLR = B.dot(-error, B.dot(reservoir.W_out, derivationConcatinationLeakingRate) + B.dot(WoutPrimeLR, B.vstack( (1, u, x))[:, 0]))
                lrGradients[t] = gradientLR
                gradientIS = B.dot(-error, B.dot(reservoir.W_out, derivationConcatinationInputScaling) + B.dot(WoutPrimeIS, B.vstack( (1, u, x))[:, 0]))
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
            reservoir.tuneSpectralRadius(reservoir.spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self.leakingRate = self.leakingRate - learningRate * gradientLR

            # update input scaling
            reservoir.tuneInputScaling(reservoir.inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            fitLoss = reservoir.fit(trainInputs, trainTargets)
            validationLoss = hlp.loss( reservoir.predict(validationInputs), validationTargets )

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if validationLoss > oldLoss:
                reservoir.tuneSpectralRadius(oldSR)
                reservoir.leakingRate = oldLR
                reservoir.tuneInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = reservoir.spectralRadius
                oldLR = reservoir.leakingRate
                oldIS = reservoir.inputScaling
                oldLoss = validationLoss

                spectralRadiuses.append(reservoir.spectralRadius)
                leakingRates.append(reservoir.leakingRate)
                inputScalings.append(reservoir.inputScaling)

                fitLosses.append(fitLoss)
                validationLosses.append(validationLoss)



        return (validationLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)


    def optimizePenaltyForEvaluationError(self, reservoir, trainInputs, trainTargets, validationInputs, validationTargets, trainLength, optimizationLength, learningRate=0.0001, epochs=1, penalty=0.1, mode="prediction"):

        Ytarget = trainTargets[None,:]
        penalty = penalty

        fitLosses = list()
        validationLosses = list()
        penalties = list()

        penaltyDerivatives = B.zeros(optimizationLength)
        oldPenalty = penalty

        reservoir.fit(trainInputs, trainTargets)
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
                predictionPoint = B.dot(reservoir.W_out, evaluationEchoFunction[:, t].reshape(-1, 1))
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