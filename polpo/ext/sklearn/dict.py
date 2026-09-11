from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing.dict import DictToValuesList, ValuesListToDict


class BiDictToValuesList(FunctionTransformer):
    def __init__(self):
        super().__init__(
            func=DictToValuesList(),
            inverse_func=ValuesListToDict(),
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.keys = X.keys()
        return self
