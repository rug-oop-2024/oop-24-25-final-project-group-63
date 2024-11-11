from autoop.core.ml.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """
    A class in which a multiple linear regression model is implemented. It
    does not have new parameters, besides the inherited one (_parameters).

    Args:
        _parameters: * definition found in "base_model.py" file *
    """

    def fit(self, observations: np.array, ground_truth: np.array) -> None:
        """
        This method trains on the given data. This implements a supervised
        learning approach and it trains the observations based on the
        ground_truth. This occurs due to the linear regression formula. The fit
        function just calculates the slopes.

        Args:
            observations: the data that is "fed" to the model.
            ground_truth: the right outcome of the observations.

        Return:
            None type. This function just trains the model.
        """
        observations = np.append(observations, np.ones((len(observations), 1)),
                                 axis=1)
        observations_transpose = observations.T
        inverse = np.linalg.inv(observations_transpose @ observations)
        slopes = inverse @ observations_transpose @ ground_truth
        self._parameters = {'slopes': slopes}

    def predict(self, observations: np.array) -> np.array:
        """
        This method uses the calculated slopes from the fit
        method and predicts the data, calculating a matrix multiplication.

        Args:
            observations: the data that has to be used in order to predict.

        Return:
            The method returns the predicted values.

        Raises:
            If the method is called and the fit function has not been called
        before, the data cannot be predicted since there is no trained data.
        """
        if not self._parameters:
            raise ValueError("The model has not been fitted yet.")

        observations = np.append(observations, np.ones((len(observations), 1)),
                                 axis=1)
        y_hat = observations @ self._parameters["slopes"]
        self._parameters["predictions"] = y_hat
        return y_hat
