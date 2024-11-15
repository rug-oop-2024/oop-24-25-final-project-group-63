from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


CLASSIFICATION_METRICS = [
    "accuracy",
    "recall",
    "macro_average",
]

REGRESSION_METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "R_squared"
]


def get_metric(name: str) -> Callable:
    """
    A factory function that returns the selected metric type if exists.

    Raise:
        NotImplementedError: The metric passed to this function does not exist
        it is not implemented.
    """
    match name:
        case "mean_squared_error":
            return MeanSquaredError()
        case "mean_absolute_error":
            return MeanAbsoluteError()
        case "accuracy":
            return Accuracy()
        case "recall":
            return Recall()
        case "macro_average":
            return MacroAverage()
        case "R_squared":
            return R2Score()
        case _:
            raise NotImplementedError("This type of metric" +
                                      " evaluation is not in the list!")


class Metric(ABC):
    """
    Base class for all metrics. It inherits from ABC class.
    """
    def __init__(self, predictions: np.array = np.array([]),
                 ground_truth: np.array = np.array([])) -> None:
        """
        A constructor method with default settings as empty arrays.

        args:
            ground_truth: The data that is correct and is used for training.
            predictions: The predictions of the data.
        """
        super().__init__()
        self._ground_truth = ground_truth
        self._predictions = predictions

    @abstractmethod
    def __call__(self, predictions: np.array,
                 ground_truth: np.array) -> None:
        pass

    def evaluate(self, predictions: np.array, ground_truth: np.array) -> float:
        """
        Evaluates the metric using the __call__ method for consistency.
        """
        return self.__call__(ground_truth, predictions)


class Accuracy(Metric):
    """
    A classification metric that returns the accuracy of the model based on
    the predictions and on the ground truth.
    """
    def __call__(self, predictions: np.array,
                 ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        correct = 0
        for gt_element, pred_element in zip(ground_truth, predictions):
            try:
                found = True
                for gt_value, pred_value in zip(gt_element, pred_element):
                    if gt_value != pred_value and found:
                        found = False
                        continue
                if found:
                    correct += 1
            except Exception:
                if gt_element == pred_element:
                    correct += 1
        return correct / len(ground_truth)


class MeanSquaredError(Metric):
    """
    A regression metric that returns the mean squared error of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, predictions: np.array,
                 ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        return np.mean((ground_truth - predictions) ** 2)


class MeanAbsoluteError(Metric):
    """
    A regression metric that returns the mean absolute error of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, predictions: np.array,
                 ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        return np.mean(np.abs(ground_truth - predictions))


class Recall(Metric):
    """
    A classification metric that returns the proportion of the correct
    predictions based on the ground truth.
    """

    def __call__(self, predictions: np.array, ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the" +
                             " same length.")

        true_positive = np.sum((predictions == 1) & (ground_truth == 1))
        false_negative = np.sum((predictions == 0) & (ground_truth == 1))

        if true_positive + false_negative > 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0


class MacroAverage(Metric):
    """
    A classification metric that returns the macro-average of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, predictions: np.array, ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        classes = np.unique(ground_truth)
        precision_per_class = []

        for element in classes:
            true_positive = np.sum((ground_truth == element) and
                                   (predictions == element))
            predicted_positive = np.sum(predictions == element)

            if predicted_positive != 0:
                precision = true_positive / predicted_positive
            else:
                precision = 0.0

            precision_per_class.append(precision)

        return np.mean(precision_per_class)


class R2Score(Metric):
    """
    A regression metric that returns the R squared statistic of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, predictions: np.array, ground_truth: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        total_variance = np.var(ground_truth)
        unexplained_variance = np.var(ground_truth - predictions)
        if total_variance > 0:
            return 1 - (unexplained_variance / total_variance)
        else:
            return 0.0
