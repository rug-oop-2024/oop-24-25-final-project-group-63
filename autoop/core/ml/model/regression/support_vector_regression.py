from sklearn.svm import SVR
from autoop.core.ml.model import Model
import numpy as np


class SVRModel(Model):
    """
    The class that wraps the support vector regression from sklearn.
    It inherits from Model.

    args:
        _parameters: * definition given in Model class *
    """
    def fit(self, observtions: np.array, ground_truth: np.array) -> None:
        """
        This method is implemented based on the blueprint of the Model class.
        It takes as parameters the observations and the ground truth of the
        dataset. It fits the data and prepares it for the prediction method.
        """
        model = SVR()
        self._parameters = {"model": model.fit(observtions, ground_truth)}

    def predict(self, observtions: np.array) -> np.array:
        """
        Based on the blueprint of the Model class, this method takes as
        parameters only the observations that has to predict based on the
        previous fit method. It returns the an array of the predicted data.
        """
        return self._parameters["model"].predict(observtions)

    def evaluate(self, observtions: np.array,
                 ground_truth: np.array) -> np.array:
        pass
