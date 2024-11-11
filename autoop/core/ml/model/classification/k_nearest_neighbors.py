import numpy as np
from collections import Counter
from autoop.core.ml.model import Model
from pydantic import field_validator


class KNearestNeighbors(Model):
    def __init__(self, k: int = 3) -> None:
        """Constructor function that sets the value for k and inheits from
        Model."""
        super().__init__()
        self._k = k

    @field_validator("k")
    def k_greater_than_zero(cls, value: int) -> None:
        """
        This method checks if the attribute k is higher than 0.
        """
        if value <= 0:
            raise ValueError("k must be greater than 0.")
        if value != int(value):
            raise ValueError("k must be an integer.")

    def fit(self, observations: np.array, ground_truth: np.array) -> None:
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
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
            }

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

        predictions = [self._predict_single(x) for x in observations]
        self._parameters["predictions"] = predictions
        return np.array(predictions)

    def _predict_single(self, observations: np.array) -> list:
        """
        This method calculates the Euclidean distance between the given data
        and all the previous stored data.

        Args:
            observations: piece of data from the data set.
        """
        distances = np.linalg.norm(
            observations - self._parameters["observations"], axis=1
        )
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[: self._k]
        k_nearest_labels = [self._parameters["ground_truth"][i]
                            for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
