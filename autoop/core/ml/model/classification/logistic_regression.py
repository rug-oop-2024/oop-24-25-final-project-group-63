from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model import Model
import numpy as np


class LogisticRegressionModel(Model):
    """
    This class wraps the logistic regression model from sklearn.
    It inherits from Model class.

    args:
        _parameters: * definition given in Model class *
    """
    def fit(self, observations: np.array, ground_truth: np.array) -> None:
        """
        Based on the blueprint of the Model class, this method takes as
        parameters the observations and the ground truth of a dataset and
        trains based on them.
        """
        model = LogisticRegression()
        model.fit(observations, ground_truth)
        self._parameters = {"model": model}

    def predict(self, observations: np.array) -> np.array:
        """
        Based on the blueprint of the Model class, this method takes as
        parameters only the observations of a dataset and predicts the
        outcome based on previously used fit method. It returns an array
        with the predicted values.
        """
        return self._parameters["model"].predict(observations)

    def evaluate(self, observations: np.array, ground_truth: np.array):
        pass
