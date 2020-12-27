# easyesn
`easyesn` is a library for recurrent neural networks using echo state networks (`ESN`s,  also called `reservoir computing`) with a high level easy to use API that is closely oriented towards `sklearn`. It aims to make the use of `ESN` as easy as possible, by providing algorithms for automatic gradient based hyperparameter tuning (of ridge regression penalty, spectral radius, leaking rate and feedback scaling), as well as transient time estimation. Furtheremore, it incorporates the ability to use spatio temporal `ESN`s for prediction and classification of geometrically extended input signals. 

`easyesn` can either use the CPU or make use of the GPU thanks to `cupy`. At the moment the use of the CPU is recommended though!

This project is based on research results for the gradient based hyperparameter optimization and transient time estimation of Luca Thiede and the spatio temporal prediction and classification techniques of Roland Zimmermann.

# Getting started

## Installation
The `easyesn` library is built using `python 3`. You cannot use it in a `python 2.x` environment. The recommended way to install `easyesn` at the moment is via `pip`. Nevertheless, you can also install `easyesn` on your own without `pip`.

### pip
You can install `easyesn` via `pip` by executing
```
pip install easyesn
```  
from a valid `python 3.x` environment, which will automatically install `easyesn` and its dependencies.

### manually
To install the library without `pip`, there are four steps you have to perform: 
1. Go to the `pypi` [page](https://pypi.python.org/pypi/easyesn) of `easyesn` and download the latest version as a `*.tar.gz` archive.
2. Extract the archive.
3. Open your command line/terminal and cd into the directory containing the `setup.py`.
4. Execute `python setup.py install` to start the installation.

## First steps
As already mentioned, the API is very similar to the one of sklearn, which makes it as easy as possible for beginners. 
For every task there is a specialized module, e.g. `ClassificationESN` for the classification of input signals, `PredictionESN` for the prediction or generation (that is a several step ahead prediction by always feeding the previous prediction back in), `RegressionESN` for mapping a signal to a real number, or `SpatioTemporalESN` for the prediction of geometrically extended input signals (for example the electric excitation on the heart surface or video frames).

Import the module typing
```python
from easyesn import PredictionESN
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

## Backends
As already mentioned in the beginning, `easyesn` can be used either on the CPU or on the GPU. To achieve this, all low level calculations are outsourced into  a backend (similiar to the backend technology of `keras`). To change the `backend` to another backend named `backendName`, there are currently two ways:

1. Modify the settings file, stored at `~/.easyesn/easyesn.json` to contain to look like this:
    ```json
    {
        "backend": "backendName"
    }
    ``` 
    and use `easyesn` without any further modification inside your code.

2. Set the `EASYESN_BACKEND` environment variable to `backendName` and use `easyesn` without any further modification inside your code.

At the moment, these are supported backend names:

| backend name | backend type | Notes |
| ------------ |:------------:|:-----:|
|   `numpy`    | `numpy` (CPU)| |
|   `np`    | `numpy` (CPU)| |
|   `cupy`    | `cupy` (GPU)| no eig & arange function which is limiting the speed |
|   `cp`    | `cupy` (GPU)| no eig & arange function which is limiting the speed |
|   `torch`    | `torch` (CPU/GPU)| **experimental** (Blasting fast but tested/developed for only on PredictionESN)|

To set which device the `torch` backend should use, use the following `easyesn.json` config:
   ```json
   {
    "backend": "torch",
    "backend_config": {
        "device": "cpu"
       }
   }
   ```
where `cpu` can be replaced with any valid `torch.device`, e.g. `cuda`.

# Notes
As of right now, the `GradientOptimizer` does not fully work - we are looking into this and try to fix the issue.

# Documentation

# Develop

## Todo
At the moment the `easyesn` library covers not only all basic features of reservoir computing but also some new, state of the art methods for its application. Nevertheless, there are still some more things which should be implemented in future versions. In the following, these feature requests and ideas are listed together which their progress:

- Ensemble ESNs (25%)
- Adding support for deep learning methods as the output method (still open)
- Improved GPU computing performance (still partially open, predictionESN done)
