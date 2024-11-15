# from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod


class Model(ABC):
    """
    An abstract base class for machine learning models. This class provides a
    blueprint for all the machine learning models that fit and predict.

    Args:
    _parameters: the dictionary that holds the "fit" and "predict" values
    """
    def __init__(self) -> None:
        """
        The constructor in which we added the private attribute and we inherit
        from both ABC and BaseModel. It does not return anything.
        """
        super().__init__()
        self._parameters = {}

    @abstractmethod
    def fit(self, observations: np.array, ground_truth: np.array) -> None:
        """
        Fit the model to the given data. The subclasses should implement
        the model's parameters.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.array) -> None:
        """
        Predict outcomes using the trained model. The subclassses should
        implement the model's parameters.
        """
        pass

    @property
    def parameters(self) -> dict:
        """
        A getter method. It returns a private attribute as a deepcopy that.
        """
        return deepcopy(self._parameters)
