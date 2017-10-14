import numpy as np

class OneHotEncoder(object):
    def __init__(self, classes=None, handleUnknown="error"):
        self.classes = classes
        self.handleUnknown = handleUnknown

    def fit(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] != 1:
            raise ValueError("The data has to one dimensional, so it either has to have the shape (*) or (*, 1).")

        if self.classes is None:
            #get unique classes
            self.classes = np.unique(data)


    def transform(self, data):
        result = np.zeros((len(data), len(self.classes)))
        for i in range(len(data)):
            result[i, np.searchsorted(self.classes, data[i])] = 1.0

        return result

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)