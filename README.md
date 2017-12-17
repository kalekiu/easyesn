# reservoir-computing
Easyesn is a library for recurrent neural networks using echo state networks (ESNs also calles reservoir computing) with a high level easy to use API that is closely oriented towards sklearn. It aims to make the use of ESN as easy as possible, by providing algorithms for automatic gradient based hyperparameter tuning (of ridge regression penalty, spectral radius, leaking rate and feedback scaling), as well as transient time estimation. Furtheremore it incorporates the ability to use spatio temporal ESNs for prediction and classification of geometrically extended input signals. 

Easyesn can either use the CPU or make use of the GPU thanks to cupy. At the moment the use of the CPU is recommended though!

This project is based on research results for the gradient based hyperparameter optimization and transient time estimation of Luca Thiede and the spatio temporal prediction and classification techniques of Roland Zimmermann.

# getting started
As already mentioned, the API is very similar to the one of sklearn, which makes it as easy as possible for beginners. 
For every task there is a specialized module, e.g. `ClassificationESN` for the classification of input signals, `RegressionESN` for the prediction or generation (that is a several step ahead prediction by always feeding the previous prediction back in) or `SpatioTemporalESN` for the prediction of geometrically extended input signals (for example the electric excitation on the heart surface or video frames).

Import the module typing
```python
from easyesn import RegressionESN
```
Now simply fit the esn using
```python
esn.fit(x_train, y_train, transientTime=100, verbose=1)
```
and predict by using
```python
y_test_pred = esn.predict(x_test, transientTime=100, verbose=1)
```

For automatic hyperparamter optimization import
```python
from easyesn.optimizers import GradientOptimizer
from easyesn.optimizers import GridSearchOptimizer
```
Next create a new object
```python
esn = PredictionESN(n_input=1, n_output=1, n_reservoir=500, leakingRate=0.2, spectralRadius=0.2, regressionParameters=[1e-2])
```
whith a penalty `1e-2` for the ridge regression. To optimize the hyperparameter also create an optimizer object
```python
opt = GradientOptimizer(esn, learningRate=0.001)
```
and use it with
```python
validationLosses, fitLosses, inputScalings, spectralRadiuses, leakingRates, learningRates = opt.optimizeParameterForTrainError(inputDataTraining, outputDataTraining, inputDataValidation, outputDataValidation, epochs=150, transientTime=100)
validationLosses, fitLosses, penalties = opt.optimizePenalty(inputDataTraining, outputDataTraining, inputDataValidation, outputDataValidation, epochs=150, transientTime=100)
```
More extensive examples can be found in the examples directory.

## installation
### pip

### build on your own

## first steps

# documentation

# develop

## todo
