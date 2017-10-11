import os
os.sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from easyesn import ESN

import numpy as np
import matplotlib.pyplot as plt

def mackeyglass(t_max, tau=17, delta_t = 1e-2, seed=42):
    np.random.seed(seed)

    beta = 0.1
    alpha = 0.2
    npow = 10

    dat = np.zeros(int(t_max//delta_t))
    tau_ind = int(tau // delta_t)
    dat[:tau_ind] = np.random.rand(tau_ind)

    for i in range(tau, len(dat)-1):
        #dat[i+1] = dat[i] + beta*dat[i-tau]/(1+pow(dat[i-tau], npow))-gamma*dat[i]
        dat[i+1] = dat[i] + delta_t * (alpha*dat[i-tau_ind]/(1+pow(dat[i-tau_ind], npow)) - beta*dat[i])


    dat = dat[::int(1/delta_t)]
    return dat

y = mackeyglass(10017, tau=17)[2*17+200:].reshape((-1,1))

def generation():
    y_train = y[:2000]
    y_test = y[2000:4000]

    esn = ESN(n_input=1, n_output=1, n_reservoir=500, noise_level=0.001, spectral_radius=0.47, leak_rate=0.20, random_seed=42, sparseness=0.2)
    train_acc = esn.fit(inputData=y_train[:-1], outputData=y_train[1:])
    print("training acc: {0:4f}\r\n".format(train_acc))

    y_test_pred = esn.generate(n=len(y_test), initial_input=y_train[-1])

    mse = np.mean( (y_test_pred-y_test)[:500]**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(y_test)
    print("testing mse: {0}".format(mse))
    print("testing rmse: {0:4f}".format(rmse))
    print("testing nrmse: {0:4f}".format(nrmse))

    plt.plot(y_test_pred, "g")
    plt.plot( y_test, "b")
    plt.show()

def pred(predictionHorizon):
    print("predicting x(t+{0})".format(predictionHorizon))
    #optimized for: predictionHorizon = 48
    y_train = y[:2000]
    y_test = y[2000-predictionHorizon:4000]

    #manual optimization
    #esn = ESN(n_input=1, n_output=1, n_reservoir=1000, noise_level=0.001, spectral_radius=.4, leak_rate=0.2, random_seed=42, sparseness=0.2)

    #gridsearch results
    esn = ESN(n_input=1, n_output=1, n_reservoir=1000, noise_level=0.0001, spectral_radius=1.35, leak_rate=0.7, random_seed=42, sparseness=0.2, solver="lsqr", regression_parameters=[1e-8])
    train_acc = esn.fit(inputData=y_train[:-predictionHorizon], outputData=y_train[predictionHorizon:], transient_quota = 0.2)
    print("training acc: {0:4f}\r\n".format(train_acc))

    y_test_pred = esn.predict(y_test[:-predictionHorizon])

    mse = np.mean( (y_test_pred-y_test[predictionHorizon:])[:]**2)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.var(y_test)
    print("testing mse: {0}".format(mse))
    print("testing rmse: {0:4f}".format(rmse))
    print("testing nrmse: {0:4f}".format(nrmse))

    import matplotlib
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble="\\usepackage{mathtools}")

    plt.figure(figsize=(8,5))
    plt.plot(y_test[predictionHorizon:], 'r', linestyle=":" )
    plt.plot(y_test_pred, 'b' , linestyle="--")
    plt.ylim([0.3, 1.6])
    plt.legend(['Signal $x(t)$', 'Vorhersage $x\'(t) \\approx x(t+{0})$'.format(predictionHorizon)],
          fancybox=True, shadow=True, ncol=2, loc="upper center")
    plt.xlabel("Zeit t")
    plt.ylabel("Signal")
    
    plt.show()

    return mse

def GridSearchTestForPred48():
    #first tst of the gridsearch for the pred48 task

    from GridSearch import GridSearch
    y_train = y[:8000]
    y_test = y[8000-48:]

    aa = GridSearch(param_grid={"n_reservoir": [900, 1000, 1100], "spectral_radius": [0.3, .35, 0.4, .45], "leak_rate": [.2, .25, .3]},
        fixed_params={"n_output": 1, "n_input": 1, "noise_level": 0.001, "sparseness": .2, "random_seed": 42},
        esnType=ESN)
    print("start fitting...")
    results = aa.fit(y_train[:-48], y_train[48:], [(y_test[:-48], y_test[48:])])
    print("done:\r\n")
    print(results)

    print("\r\nBest result (mse =  {0}):\r\n".format(aa._best_mse))
    print(aa._best_params)

#generation()
pred(84)