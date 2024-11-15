from autoop.core.ml.model import Model
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class GradientBoostModel(Model):
    """
    A class in which a gradient boost regression model is implemented.
    Different from the previous one, this model is using methods from
    a library. It inherits from "base_model.py".

    Args:
        _parameters: * definition found in "base_model.py" file *
    """

    def fit(self, observations: np.array, ground_truth: np.array) -> None:
        """
        A method that trains the data using "gradient boost" class. We use the
        predefined methods of that class to train the data and we add
        it in the private attribute "_parameters".

        Args:
            observations: the data that is "fed" to the model.
            ground_truth: the correct outcome of the observations.

        Return:
            It does not return anything. It just trains the data.
        """
        model = GradientBoostingRegressor()
        model.fit(observations, ground_truth)
        self._parameters["model"] = model

    def predict(self, observations: np.array) -> np.array:
        """
        A method that predicts the gradient boost regression based on the
        attribute "observations". We just calculate the prediction using
        the intercept and slopes.

        Args:
            observations: the data that has to be used in order to predict.

        Return:
            It returns the prediction based on the given data.

        Raises:
            If the method is called and the fit function has not been called
        before, the data cannot be predicted since there is no trained data.
        """
        if self._parameters is None:
            raise ValueError("Model not fitted. Call 'fit' with appropriate"
                             "arguments before using predict")
        return self._parameters["model"].predict(observations)
