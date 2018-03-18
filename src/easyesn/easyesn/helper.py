"""
    Utility methods which are used for the ESNs.
"""

import numpy as np

"""
    Calculates the mutual information between the two signals x and y.
"""
def calculate_mutualinformation(x, y, bins):
    pxy, _, _ = np.histogram2d(x, y, bins)
    px, _, = np.histogram(x, bins)
    py, _, = np.histogram(y, bins)

    pxy = pxy/np.sum(pxy)
    px = px/np.sum(px)
    py = py/np.sum(py)

    pxy = pxy[np.nonzero(pxy)]
    px = px[np.nonzero(px)]
    py = py[np.nonzero(py)]

    hxy = -np.sum(pxy*np.log2(pxy))
    hx = -np.sum(px*np.log2(px))
    hy = -np.sum(py*np.log2(py))

    MI = hx+hy-hxy

    return MI

"""
    Tries to calculate the input scaling of an ESN by using the mutual information.
"""
def calculate_esn_mi_input_scaling(input_data, output_data):
    if len(input_data) != len(output_data):
        raise ValueError("input_data and output_data do not have the same length -  {0} vs. {1}".format(len(input_data), len(output_data)))

    #Scott's rule to calculate nbins
    std_output = np.std(output_data)
    nbins = int(np.ceil(2.0/(3.5*std_output/np.power(len(input_data), 1.0/3.0))))

    mi = np.zeros(input_data.shape[1])
    for i in range(len(mi)):
        mi[i] = calculate_mutualinformation(input_data[:, i], output_data, nbins)
    scaling = mi / np.max(mi)

    return scaling


"""
    Estimates autocorrelation of given np array x
"""
def autocorrelation(self, x):
    n = x.shape[0]
    variance = B.var(x)
    x = x - B.mean(x)
    r = B.correlate(x, x, mode='full')[-n:]
    assert B.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result

"""
    Calculates SWD (sliding window difference) with the specified intervall using the first `interval` entries of the series as the window
    Returns a tuple consisting of the point of minimum and the whole SWD series
"""
def SWD(series, intervall):
    differences = np.zeros(series.shape[0] - 2 * intervall)
    reference_series = series[:intervall]
    for i in range(intervall, series.shape[0] - intervall):
        differences[i - intervall] = np.sum(np.abs(reference_series - series[i:i + intervall]))

    return np.argmin(differences) + intervall, differences


def loss(prediction, target):
    return np.mean( ( prediction - target ) ** 2 )
