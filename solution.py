"""62136"""

import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, model: pd.DataFrame):
        self.model = model

    def predict(self, vector):
        pass # TODO: return the most probable class

if __name__ == "__main__":
    df = pd.read_csv("data/house-votes-84.data")
    df.columns = ["class"] + list(map(str, range(16)))
    classifier = NaiveBayesClassifier(df)
    assert "democrat" == classifier.predict(["n","y","y","n","y","y","n","n","n","n","n","n","y","y","y","y"]), "Underfitting"