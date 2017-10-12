from numpy import *
import matplotlib.pyplot as plt

class Reservoir:

    def __init__(self, input_dim, target_dim, size, spectralRadius=0.9, feedbackScaling=1, inputScaling=1, leakingRate=0.3,
                 randomState=42, density=0.1, transientTime=0):
        random.seed(randomState)
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.size = size
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.leakingRate = leakingRate
        self.density = density
        self.transientTime = transientTime

        self.generateReservoirConnections()



    def generateReservoirConnections(self):
        # generate input matrix
        self.W_in_uniform = (random.rand(self.size, self.input_dim + 1) - 0.5)
        self.W_in = self.inputScaling * self.W_in_uniform

        # generate dense random matrix
        self.W_rand = (random.rand(self.size, self.size) - 0.5)

        # generate sparse topology matrix with either 0 or 1
        self.W_top = random.rand(self.size, self.size)
        self.W_top[where(self.W_top < 1 - self.density)] = 0
        self.W_top[where(self.W_top >= 1 - self.density)] = 1

        # multiply both for reservoir matrix and scale reservoir to spectral radius
        print("Calculate spectral radius")
        self.setTopology(self.W_top)
        print("Done")


    def setTopology(self, W_top):
        self.W_top = W_top
        self.W_uniform = W_top * self.W_rand

        # scale spectral radius
        biggestEigenvalue = max(abs(linalg.eig(self.W_uniform)[0]))
        self.W_uniform = self.W_uniform / biggestEigenvalue
        self.W = self.W_uniform * self.spectralRadius

    # rescale spectral radius
    def tuneSpectralRadius(self, newSpectralRadius):
        self.W = self.W_uniform * newSpectralRadius
        self.spectralRadius = newSpectralRadius

    # rescale input scaling
    def tuneInputScaling(self, newInputScaling):
        self.W_in = self.W_in_uniform * newInputScaling
        self.inputScaling = newInputScaling

    # rescale leaking rate
    def tuneLeakingRate(self, newLeakingRate):
        self.leakingRate = newLeakingRate

    # calculate RMSE
    def Loss(self, prediction, target):
        return sqrt(((target - prediction) ** 2).mean(axis=None))

    def ridgeRegression(self, Y, X, reg):
        X_T = X.T
        return dot(dot(Y, X_T), linalg.inv(dot(X, X_T) + reg * eye(1 + self.input_dim + self.size)))

    # get reservoir output
    def readOut(self, u, x):
        return dot(self.W_out, vstack((1, u, x)))

    # terms within nonlinearity
    def X(self, x, u):
        return dot(self.W_in, vstack((1,u))) + dot(self.W, x)

    def f(self, x):
        return tanh(x)

    #update step of the reservoir states
    def updateNeuronState(self, x, u):
        newNeuronState = self.f(self.X(x, u))
        return (1 - self.leakingRate) * x + self.leakingRate * newNeuronState



################ Fitting and predicting with reservoir ################
    def fit(self, inputs, targets, trainLength, penalty=0.1, errorEvaluationLength=500):

        # initialize
        self.designMatrix = zeros((1 + self.input_dim + self.size, trainLength - self.transientTime))
        Ytarget = targets[None, self.transientTime:trainLength]
        self.x = zeros((self.size, 1))


        for t in range(trainLength):
            u = inputs[t].reshape(-1,1)
            self.x = self.updateNeuronState(self.x, u)
            if t >= self.transientTime:
                self.designMatrix[:, t - self.transientTime] = vstack((1, u, self.x))[:, 0]

        # compute W_out via ridge regression
        self.W_out = self.ridgeRegression(Ytarget, self.designMatrix, penalty)


        # calculate fit and evaluation loss
        self.prediction = dot(self.W_out, self.designMatrix)
        fitLoss = self.Loss(self.prediction, targets[self.transientTime :trainLength ])

        EvaPrediction = self.predictOnePointAhead(errorEvaluationLength, inputs, trainLength)
        evaLoss = self.Loss(EvaPrediction, targets[trainLength :trainLength + errorEvaluationLength])
        print("Evaluation RMSE= " + str(evaLoss) + " and Fit RMSE= " + str(fitLoss))

        return (evaLoss, fitLoss)



    def predictOnePointAhead(self, predictionLength, data, lastPoint):
        prediction = zeros((self.target_dim, predictionLength))
        x = copy(self.x)

        for t in range(predictionLength):
            u = data[lastPoint + t].reshape(-1,1)
            x = self.updateNeuronState(x, u)
            predictionPoint = dot(self.W_out, vstack((1, u, x)))
            prediction[:, t] = predictionPoint


        return prediction


################################ Optimization ###################################

    # f`(X)
    def activationDerivation(self, X):
        return 4 / (2 + exp(2 * X) + exp(-2 * X))

    # del x / del alpha
    def derivationForLeakingRate(self, oldDerivative, u, x):
        a = self.leakingRate
        X = self.X(x, u)
        return (1-a) * oldDerivative - x + self.f(X) + a * self.activationDerivation(X) * dot(self.W, oldDerivative )

    # del x / del rho
    def derivationForSpectralRadius(self, oldDerivative, u, x):
        a = self.leakingRate
        X = self.X(x, u)
        return (1-a) * oldDerivative + a * self.activationDerivation(X) * ( dot(self.W, oldDerivative) + dot(self.W_uniform, x) )

    # del x / del s_in
    def derivationForInputScaling(self, oldDerivative, u, x):
        a = self.leakingRate
        X = self.X(x, u)
        u = vstack((1,u))
        return (1-a)  * oldDerivative + a * self.activationDerivation(X) * ( dot(self.W, oldDerivative) + dot(self.W_in_uniform, u) )

    # del W_out / del beta
    def derivationForPenalty(self, Y, X, penalty):
        X_T = X.T
        term2 = linalg.inv( dot( X, X_T ) + penalty * eye(1 + self.input_dim + self.size) )
        return - dot( dot( Y, X_T ), dot( term2, term2 ) )

    # del W_out / del (alpha, rho or s_in)
    def derivationWoutForP(self, Y, X, XPrime, penalty):
        X_T = X.T
        XPrime_T = XPrime.T

        # A = dot(X,X_T) + penalty*eye(1 + self.target_dim + self.size)
        # APrime = dot( XPrime, X_T) + dot( X, XPrime_T )
        # APrime_T = APrime.T
        # InvA = linalg.inv(A)
        # InvA_T = InvA.T
        #
        # term1 = dot(XPrime_T, InvA)
        #
        # term21 = -dot( InvA, dot( APrime, InvA ) )
        # term22 = dot( dot( dot( InvA, InvA_T), APrime_T), eye(1 + self.target_dim + self.size) - dot( A, InvA ) )
        # term23 = dot( dot( eye(1 + self.target_dim + self.size) - dot( InvA, A ), APrime_T), dot( InvA_T, InvA) )
        # term2 = dot( X_T, term21 + term22 + term23 )
        #
        # return dot( Y, term1 + term2)

        term1 = linalg.inv(dot(X,X_T) + penalty*eye(1 + self.input_dim + self.size))
        term2 = dot( XPrime, X_T) + dot( X, XPrime_T )

        return dot( Y, dot( XPrime_T, term1 ) - dot( dot( dot( X_T, term1 ), term2 ), term1 ) )






    def optimizeParameterForTrainError(self, inputs, targets, trainLength, learningRate = 0.0001, epochs=1, penalty=0.1, errorEvaluationLength=500):
        # initializations of arrays:
        Ytarget = targets[None, self.transientTime:trainLength]

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        fitLosses = list()
        evaLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = zeros(trainLength - self.transientTime)
        lrGradients = zeros(trainLength - self.transientTime)
        isGradients = zeros(trainLength - self.transientTime)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        lrGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        isGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))

        # initialize fallback Parameter
        oldSR = self.spectralRadius
        oldLR = self.leakingRate
        oldIS = self.inputScaling

        # initialize self.designMatrix and self.W_out
        _, oldLoss, = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=1)


        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")
            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = zeros((self.size, 1))
            derivationLeakingRate = zeros((self.size, 1))
            derivationInputScaling = zeros((self.size, 1))

            # initialize the neuron states new
            x = zeros((self.size, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t in range(trainLength):
                u = inputs[t].reshape(-1, 1)
                oldx = x
                x = self.updateNeuronState(x, u)

                # calculate the del /x del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(derivationInputScaling, u, oldx)

                if t >= self.transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    derivationConcatinationSpectralRadius = concatenate((zeros(self.input_dim + 1), derivationSpectralRadius[:, 0]), axis=0)
                    derivationConcatinationLeakingRate = concatenate((zeros(self.input_dim + 1), derivationLeakingRate[:, 0]), axis=0)
                    derivationConcatinationInputScaling = concatenate((zeros(self.input_dim + 1), derivationInputScaling[:, 0]), axis=0)

                    # add to matrix
                    srGradientsMatrix[:, t - self.transientTime] = derivationConcatinationSpectralRadius
                    lrGradientsMatrix[:, t - self.transientTime] = derivationConcatinationLeakingRate
                    isGradientsMatrix[:, t - self.transientTime] = derivationConcatinationInputScaling

            # calculate del W_out / del (rho, alpha, s_in) based on the designMatrix and the derivative of the designMatrix we just calculated
            WoutPrimeSR = self.derivationWoutForP(Ytarget, self.designMatrix, srGradientsMatrix, penalty)
            WoutPrimeLR = self.derivationWoutForP(Ytarget, self.designMatrix, lrGradientsMatrix, penalty)
            WoutPrimeIS = self.derivationWoutForP(Ytarget, self.designMatrix, isGradientsMatrix, penalty)

            # reinitialize the states
            x = zeros((self.size, 1))
            # go through the train time again, and this time, calculate del error / del (rho, alpha, s_in) based on del W_out and the single derivatives
            for t in range(trainLength):
                u = inputs[t].reshape(-1, 1)
                x = self.updateNeuronState(x, u)

                if t >= self.transientTime:
                    # calculate error at given time step
                    error = (targets[t] - self.readOut(u, x)).T

                    # calculate gradients
                    gradientSR = dot(-error, dot( WoutPrimeSR, vstack((1, u, x))[:, 0] ) + dot(self.W_out, srGradientsMatrix[:,t-self.transientTime]) )
                    srGradients[t - self.transientTime] = gradientSR

                    gradientLR = dot(-error, dot( WoutPrimeLR, vstack((1, u, x))[:, 0] ) + dot(self.W_out, lrGradientsMatrix[:,t-self.transientTime]) )
                    lrGradients[t - self.transientTime] = gradientLR

                    gradientIS = dot(-error, dot( WoutPrimeIS, vstack((1, u, x))[:, 0] ) + dot(self.W_out, isGradientsMatrix[:,t-self.transientTime]) )
                    isGradients[t - self.transientTime] = gradientIS


            # sum up the gradients del error / del (rho, alpha, s_in) to get final gradient
            gradientSR = sum(srGradients)
            gradientLR = sum(lrGradients)
            gradientIS = sum(isGradients)


            # normalize gradients to length 1
            gradientVectorLength = sqrt(gradientSR ** 2 + gradientLR ** 2 + gradientIS ** 2)
            #gradientVectorLength = sqrt(gradientSR ** 2 + gradientLR ** 2)
            #gradientVectorLength = sqrt(gradientSR ** 2)

            gradientSR /= gradientVectorLength
            gradientLR /= gradientVectorLength
            gradientIS /= gradientVectorLength

            # update spectral radius
            self.tuneSpectralRadius(self.spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self.leakingRate = self.leakingRate - learningRate * gradientLR

            # update input scaling
            self.tuneInputScaling(self.inputScaling - learningRate * gradientIS)

            # calculate the errors and update the self.designMatrix and the W_out
            evaLoss, fitLoss = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=errorEvaluationLength)

            if fitLoss > oldLoss:
                self.tuneSpectralRadius(oldSR)
                self.leakingRate = oldLR
                self.tuneInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = self.spectralRadius
                oldLR = self.leakingRate
                oldIS = self.inputScaling
                oldLoss = fitLoss

                spectralRadiuses.append(self.spectralRadius)
                leakingRates.append(self.leakingRate)
                inputScalings.append(self.inputScaling)

                fitLosses.append(fitLoss)
                evaLosses.append(evaLoss)


        evaLoss = evaLosses[-1]
        fitLoss = fitLosses[-1]

        return (evaLoss, fitLoss, evaLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)



####################################################################################################################################################


    def optimizeParameterForEvaluationError(self, inputs, targets,  trainLength, optimizationLength, learningRate = 0.0001, epochs=1, penalty=0.1):
        # initializations of arrays:
        Ytarget = targets[None, self.transientTime:trainLength]

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        fitLosses = list()
        evaLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = zeros(optimizationLength)
        lrGradients = zeros(optimizationLength)
        isGradients = zeros(optimizationLength)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        lrGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        isGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))

        # initialize variables for the "when the error goes up, go back and divide learning rate by 2" mechanism
        oldSR = self.spectralRadius
        oldLR = self.leakingRate
        oldIS = self.inputScaling

        # initialize self.designMatrix and self.W_out
        oldLoss, _, = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)

        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = zeros((self.size, 1))
            derivationLeakingRate = zeros((self.size, 1))
            derivationInputScaling = zeros((self.size, 1))
            x = zeros((self.size, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t in range(trainLength):
                u = inputs[t].reshape(-1, 1)
                oldx = x
                x = self.updateNeuronState(x, u)

                # calculate the del x/ del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(derivationInputScaling, u, oldx)

                if t >= self.transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    srGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationSpectralRadius[:, 0]), axis=0)
                    lrGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationLeakingRate[:, 0]), axis=0)
                    isGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationInputScaling[:, 0]), axis=0)


            # add to matrix
            WoutPrimeSR = self.derivationWoutForP(Ytarget, self.designMatrix, srGradientsMatrix, penalty)
            WoutPrimeLR = self.derivationWoutForP(Ytarget, self.designMatrix, lrGradientsMatrix, penalty)
            WoutPrimeIS = self.derivationWoutForP(Ytarget, self.designMatrix, isGradientsMatrix, penalty)

            # this time go through validation length
            for t in range(optimizationLength):
                u = inputs[t + trainLength].reshape(-1, 1)
                oldx = x
                x = self.updateNeuronState(x, u)

                # calculate error at given time step
                error = ( targets[t + trainLength] - self.readOut(u, x) ).T

                # calculate del x / del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(derivationInputScaling, u, oldx)

                # concatenate derivations with 0
                derivationConcatinationSpectralRadius = concatenate((zeros(self.input_dim + 1), derivationSpectralRadius[:, 0]), axis=0)
                derivationConcatinationLeakingRate = concatenate((zeros(self.input_dim + 1), derivationLeakingRate[:, 0]), axis=0)
                derivationConcatinationInputScaling = concatenate((zeros(self.input_dim + 1), derivationInputScaling[:, 0]), axis=0)

                # calculate gradients
                gradientSR = dot(-error, dot(self.W_out, derivationConcatinationSpectralRadius) + dot( WoutPrimeSR, vstack((1, u, x))[:, 0] ) )
                srGradients[t] = gradientSR

                gradientLR = dot(-error, dot(self.W_out, derivationConcatinationLeakingRate) + dot( WoutPrimeLR, vstack((1, u, x))[:, 0] ) )
                lrGradients[t] = gradientLR

                gradientIS = dot(-error, dot(self.W_out, derivationConcatinationInputScaling) + dot( WoutPrimeIS, vstack((1, u, x))[:, 0] ) )
                isGradients[t] = gradientIS


            # sum up the gradients del error / del (rho, alpha, s_in) to get final gradient
            gradientSR = sum(srGradients)
            gradientLR = sum(lrGradients)
            gradientIS = sum(isGradients)

            # normalize length of gradient to 1
            gradientVectorLength = sqrt(gradientSR ** 2 + gradientLR ** 2 + gradientIS ** 2)
            #gradientVectorLength = sqrt(gradientIS ** 2 + gradientLR ** 2 )

            gradientSR /= gradientVectorLength
            gradientLR /= gradientVectorLength
            gradientIS /= gradientVectorLength

            # update spectral radius
            self.tuneSpectralRadius(self.spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self.leakingRate = self.leakingRate - learningRate * gradientLR

            # update input scaling
            self.tuneInputScaling(self.inputScaling - learningRate * gradientIS)


            # calculate the errors and update the self.designMatrix and the W_out
            evaLoss, fitLoss = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if evaLoss > oldLoss:
                self.tuneSpectralRadius(oldSR)
                self.leakingRate = oldLR
                self.tuneInputScaling(oldIS)
                learningRate = learningRate / 2
            else:
                oldSR = self.spectralRadius
                oldLR = self.leakingRate
                oldIS = self.inputScaling
                oldLoss = evaLoss

                spectralRadiuses.append(self.spectralRadius)
                leakingRates.append(self.leakingRate)
                inputScalings.append(self.inputScaling)

                fitLosses.append(fitLoss)
                evaLosses.append(evaLoss)



        evaLoss = evaLosses[-1]
        fitLoss = fitLosses[-1]

        return (evaLoss, fitLoss, evaLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses)


    def optimizePenaltyForEvaluationError(self, inputs, targets, trainLength, optimizationLength, learningRate = 0.0001, epochs=1, penalty=0.1):

        Ytarget = targets[None, self.transientTime:trainLength]
        penalty = penalty

        fitLosses = list()
        evaLosses = list()
        penalties = list()

        penaltyDerivatives = zeros(optimizationLength)

        oldPenalty = penalty

        oldLoss, fitLoss = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)

        evaluationEchoFunction = zeros((1+self.size+self.input_dim, optimizationLength))
        u = inputs[trainLength]
        x = self.x
        for t in range(optimizationLength):
            x = self.updateNeuronState(x, u)
            u = inputs[trainLength + t].reshape(-1, 1)
            evaluationEchoFunction[:,t] = vstack((1, u, x)).squeeze()
            # u = predictionPoint

        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            penaltyDerivative = self.derivationForPenalty(Ytarget, self.designMatrix, penalty)

            for t in range(optimizationLength):
                predictionPoint = dot(self.W_out, evaluationEchoFunction[:,t].reshape(-1,1))
                error = ( targets[trainLength + t] - predictionPoint ).T
                penaltyDerivatives[t] = - dot( dot(error, penaltyDerivative), vstack((1, u, x))[:, 0] )
                u = inputs[trainLength + t].reshape(-1, 1)
                #u = predictionPoint

            penaltyGradient = sum(penaltyDerivatives)
            penaltyGradient /= sqrt(penaltyGradient**2)

            penalty = penalty - learningRate * penaltyGradient

            evaLoss, fitLoss = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)


            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if evaLoss > oldLoss:
                penalty = oldPenalty
                learningRate = learningRate / 2
            else:
                oldPenalty = penalty
                oldLoss = evaLoss

                fitLosses.append( fitLoss )
                evaLosses.append( evaLoss )
                penalties.append( penalty )

        return (evaLoss, fitLoss, evaLosses, fitLosses, penalties)



    def optimizeAllParameter(self, inputs, targets,  trainLength, optimizationLength, learningRate = 0.01, learningRatePenalty=0.0001, epochs=1, penalty=0.1):
        # initializations of arrays:
        Ytarget = targets[None, self.transientTime:trainLength]

        # initializations for plotting parameter and losses at the end
        inputScalings = list()
        leakingRates = list()
        spectralRadiuses = list()
        penalties = list()
        fitLosses = list()
        evaLosses = list()

        # initializations for arrays which collect all the gradients of the error of the single time steps, which get add at the end
        srGradients = zeros(optimizationLength)
        lrGradients = zeros(optimizationLength)
        isGradients = zeros(optimizationLength)
        penaltyDerivatives = zeros(optimizationLength)

        # collecting the single derivatives  - > this is the derivation of design matrix when filled
        srGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        lrGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))
        isGradientsMatrix = zeros((self.size + self.input_dim + 1, trainLength - self.transientTime))

        # initialize variables for the "when the error goes up, go back and divide learning rate by 2" mechanism
        oldSR = self.spectralRadius
        oldLR = self.leakingRate
        oldIS = self.inputScaling
        oldPenalty = penalty

        # initialize self.designMatrix and self.W_out
        oldLoss, _ = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)

        for epoch in range(epochs):
            print("###################### Start epoch: " + str(epoch) + " ##########################")

            # initialize del x / del (rho, alpha, s_in) and reservoir state itself
            derivationSpectralRadius = zeros((self.size, 1))
            derivationLeakingRate = zeros((self.size, 1))
            derivationInputScaling = zeros((self.size, 1))
            x = zeros((self.size, 1))

            # go thorugh the train length (e.g. the time, on which W_out gets calculated)
            for t in range(trainLength):
                u = inputs[t].reshape(-1, 1)
                oldx = x
                x = self.updateNeuronState(x, u)

                # calculate the del x/ del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(derivationInputScaling, u, oldx)

                if t >= self.transientTime:

                    # concatenate with zeros (for the derivatives of the input and the 1, which are always 0)
                    srGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationSpectralRadius[:, 0]), axis=0)
                    lrGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationLeakingRate[:, 0]), axis=0)
                    isGradientsMatrix[:, t - self.transientTime] = concatenate((zeros(self.input_dim + 1), derivationInputScaling[:, 0]), axis=0)


            # add to matrix
            WoutPrimeSR = self.derivationWoutForP(Ytarget, self.designMatrix, srGradientsMatrix, penalty)
            WoutPrimeLR = self.derivationWoutForP(Ytarget, self.designMatrix, lrGradientsMatrix, penalty)
            WoutPrimeIS = self.derivationWoutForP(Ytarget, self.designMatrix, isGradientsMatrix, penalty)

            # penaltyDerivation
            penaltyDerivative = self.derivationForPenalty(Ytarget, self.designMatrix, penalty)

            # this time go through validation length
            for t in range(optimizationLength):
                u = inputs[t + trainLength].reshape(-1, 1)
                oldx = x
                x = self.updateNeuronState(x, u)

                # calculate error at given time step
                error = ( targets[t + trainLength] - self.readOut(u, x) ).T

                # calculate del x / del (rho, alpha, s_in)
                derivationSpectralRadius = self.derivationForSpectralRadius(derivationSpectralRadius, u, oldx)
                derivationLeakingRate = self.derivationForLeakingRate(derivationLeakingRate, u, oldx)
                derivationInputScaling = self.derivationForInputScaling(derivationInputScaling, u, oldx)

                # concatenate derivations with 0
                derivationConcatinationSpectralRadius = concatenate((zeros(self.input_dim + 1), derivationSpectralRadius[:, 0]), axis=0)
                derivationConcatinationLeakingRate = concatenate((zeros(self.input_dim + 1), derivationLeakingRate[:, 0]), axis=0)
                derivationConcatinationInputScaling = concatenate((zeros(self.input_dim + 1), derivationInputScaling[:, 0]), axis=0)

                # calculate gradients
                gradientSR = dot(-error, dot(self.W_out, derivationConcatinationSpectralRadius) + dot( WoutPrimeSR, vstack((1, u, x))[:, 0] ) )
                srGradients[t] = gradientSR

                gradientLR = dot(-error, dot(self.W_out, derivationConcatinationLeakingRate) + dot( WoutPrimeLR, vstack((1, u, x))[:, 0] ) )
                lrGradients[t] = gradientLR

                gradientIS = dot(-error, dot(self.W_out, derivationConcatinationInputScaling) + dot( WoutPrimeIS, vstack((1, u, x))[:, 0] ) )
                isGradients[t] = gradientIS

                penaltyDerivatives[t] = - dot(dot(error, penaltyDerivative), vstack((1, u, x))[:, 0])


            # sum up the gradients del error / del (rho, alpha, s_in) to get final gradient
            gradientSR = sum(srGradients)
            gradientLR = sum(lrGradients)
            gradientIS = sum(isGradients)
            penaltyGradient = sum(penaltyDerivatives)

            # normalize length of gradient to 1
            gradientVectorLength = sqrt(gradientSR ** 2 + gradientLR ** 2 + gradientIS ** 2)
            penaltyGradientVectorLength = sqrt(penaltyGradient**2)

            gradientSR /= gradientVectorLength
            gradientLR /= gradientVectorLength
            gradientIS /= gradientVectorLength
            penaltyGradient /= penaltyGradientVectorLength


            # update spectral radius
            self.tuneSpectralRadius(self.spectralRadius - learningRate * gradientSR)

            # update leaking rate
            self.leakingRate = self.leakingRate - learningRate * gradientLR

            # update input scaling
            self.tuneInputScaling(self.inputScaling - learningRate * gradientIS)

            # update penalty
            penalty = penalty - learningRatePenalty * penaltyGradient


            # calculate the errors and update the self.designMatrix and the W_out
            evaLoss, fitLoss = self.fit(inputs, targets, trainLength, penalty=penalty, errorEvaluationLength=optimizationLength)

            # this is the "when the error goes up, go back and divide learning rate by 2" mechanism
            if evaLoss > oldLoss:
                self.tuneSpectralRadius(oldSR)
                self.leakingRate = oldLR
                self.tuneInputScaling(oldIS)
                penalty = oldPenalty
                learningRate = learningRate / 2
                learningRatePenalty = learningRatePenalty / 6
            else:
                oldSR = self.spectralRadius
                oldLR = self.leakingRate
                oldIS = self.inputScaling
                oldPenalty = penalty
                oldLoss = evaLoss

                spectralRadiuses.append(self.spectralRadius)
                leakingRates.append(self.leakingRate)
                inputScalings.append(self.inputScaling)
                penalties.append(penalty)

                fitLosses.append(fitLoss)
                evaLosses.append(evaLoss)



        evaLoss = evaLosses[-1]
        fitLoss = fitLosses[-1]

        return (evaLoss, fitLoss, evaLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses, penalties)




# ### example usage ###
# import numpy as np
# def generateData(T):
#     x = np.arange(0,T*0.1,0.1)
#     return np.sin(x) + np.sin(0.51*x) +np.sin(0.22*x) + np.sin(0.1002*x) + np.sin(0.05343*x)#np.sin(x)
#
# data = generateData(10000)
#
# data = data/np.max(data)
# targetData = np.roll(data,-1)
#
# plt.plot(data)
# plt.xlim(0,3000)
# plt.xlabel("t")
# plt.ylabel("f(t)")
# plt.show()
#
#
# transientTime = 1500
# trainTime = 2500
# evaTime = 200
# epocsh = 60
#
# reservoir = Reservoir(1,1,20, density=0.2, spectralRadius=0.5, leakingRate=0.5, inputScaling=1, transientTime=transientTime)
#
# print("#################################################")
# evaLoss, fitLoss, evaLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses, penalties = reservoir.optimizeParameterForTrainError(data, targetData, trainTime, epochs = epocsh, learningRate = 0.015, errorEvaluationLength=evaTime, penalty=0.005)
# #evaLoss, fitLoss, evaLosses, fitLosses, inputScalings, leakingRates, spectralRadiuses, penalties = reservoir.optimizeParameterForEvaluationError(data, targetData, trainTime, evaTime, epochs = epocsh, learningRate = 0.02, penalty=0.005)
# print("##################################################")
#
#
#
# ### plot results ###
# plt.plot(inputScalings)
# plt.title("InputScaling")
# plt.show()
# plt.plot(leakingRates)
# plt.title("LeakingRate")
# plt.show()
# plt.plot(spectralRadiuses)
# plt.title("SpectralRadius")
# plt.show()
# plt.plot(fitLosses)
# plt.title("Fit error")
# plt.show()
# plt.plot(evaLosses)
# plt.title("Evaluation Error")
# plt.show()
# plt.plot(penalties)
# plt.title("Penalties")
# plt.show()
#
#
# plt.plot(np.dot(reservoir.W_out, reservoir.designMatrix)[0])
# plt.title("Fit")
# plt.show()
# plt.plot(reservoir.designMatrix.T)
# plt.title("Echo functions")
# plt.show()
# plt.plot(targetData[trainTime:trainTime + evaTime], label="data")
# plt.plot(reservoir.predictOnePointAhead(evaTime, data, trainTime)[0], label="prediction")
# plt.legend()
# plt.show()
