import numpy as np
from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(Model):
    def fit(self, observations: np.array,
            ground_truth: np.array) -> None | ValueError:
        """
        This method trains the model on the given data and checks it
        with the correct answer. This just adds the values into the
        dictionary and it's considered trained data.

        Args:
            observations: the data that is "fed" to the model.
            ground_truth: the right outcome of the observations.

        Return:
            It does not return anything. It just trains on the data.
        """
        if len(observations) < 5:
            raise ValueError("There is not enough data to clasify properly." +
                             " Please add a bigger dataset or change" +
                             " the scale of observations to more. It" +
                             " needs at least 5 rows!")
        model = KNeighborsClassifier()
        model.fit(observations, ground_truth)
        self._parameters["model"] = model

    def predict(self, observations: np.array) -> np.array:
        """
        Description:
            This method predicts the outcome of the given data. This works by
        calculating the distance for each instance from the given data and
        it takes the closest k neighbors.

        Args:
            observations: the data that has to be used to predict.

        Raises:
            If the method "predict" is used before calling the "fit" method,
            it will not work because there is not trained data to make the
            prediction on.
        """
        if self._parameters is None:
            raise ValueError(
                "Model not fitted. Call fit with appropriate"
                "arguments before using predict"
            )
        return self._parameters["model"].predict(observations)
