
class Pipeline(object):
    def __init__(self, **items):
        self.items = items

    def fit(self, trainingInput, trainingOutput, validationInput, validationOutput, verbose=1):
        for item in self.items:
            item.fit(trainingInput, trainingOutput, validationInput, validationOutput, verbose)