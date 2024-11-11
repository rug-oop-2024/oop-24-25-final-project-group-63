from autoop.core.ml.model import Model
from sklearn.linear_model import Lasso as Las
import numpy as np


class Lasso(Model):
    """
    A class in which a multiple linear regression model is implemented.
    Different from the previous one, this model is using methods from
    a library. It inherits from "base_model.py".

    Args:
        _parameters: * definition found in "base_model.py" file *
    """

    def fit(self, observations: np.array, ground_truth: np.array) -> None:
        """
        A method that trains the data using "Lasso" class. We use the
        predefined methods of that class to train the data and we add
        it in the private attribute "_parameters".

        Args:
            observations: the data that is "fed" to the model.
            ground_truth: the right outcome of the observations.

        Return:
            It does not return anything. It just trains the data.
        """
        lasso = Las()
        lasso.fit(observations, ground_truth)
        self._parameters = {"Fitted model": lasso}

    def predict(self, observations: np.array) -> np.array:
        """
        A method that predicts the multiple linear regression based on the
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

        y_hat = self._parameters["Fitted model"].predict(observations)
        return y_hat