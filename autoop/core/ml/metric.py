from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


METRICS = {
    "mean_squared_error",
    "mean_absolute_error",
    "accuracy",
    "cohens_kappa",
    "macro_average",
    "R_squared"
}


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
        case "cohens_kappa":
            return CohensKappa()
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
    def __init__(self, ground_truth: np.array = np.array([]),
                 predictions: np.array = np.array([])) -> None:
        """
        A constructor method with default settings as empty arrays.

        args:
            _ground_truth: The data that is correct and is used for training.
            _predictions: The predictions of the data.
        """
        self._ground_truth = ground_truth
        self._predictions = predictions

    @abstractmethod
    def __call__(self, ground_truth: np.array,
                 predictions: np.array) -> None:
        pass

    def evaluate(self, ground_truth: np.array, predictions: np.array) -> float:
        """
        Evaluates the metric using the __call__ method for consistency.
        """
        return self.__call__(ground_truth, predictions)


class Accuracy(Metric):
    """
    A classification metric that returns the accuracy of the model based on
    the predictions and on the ground truth.
    """
    def __call__(self, ground_truth: np.array,
                 predictions: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        correct = np.sum(ground_truth == predictions)
        return correct / len(ground_truth)


class MeanSquaredError(Metric):
    """
    A regression metric that returns the mean squared error of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, ground_truth: np.array,
                 predictions: np.array) -> float:
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
    def __call__(self, ground_truth: np.array,
                 predictions: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        i = 0
        abs_sum = 0
        while i < len(predictions):
            abs_sum = abs_sum + abs(ground_truth[i] - predictions[i])
            i += 1
        return abs_sum / len(predictions)


class CohensKappa(Metric):
    """
    A classification metric that returns the Cohen's Kappa value of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, ground_truth: np.array, predictions: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("ground_truth and predictions must" +
                             " be the same length.")

        unique_labels = np.unique(np.concatenate((ground_truth, predictions)))
        label_to_index = {label: idx for idx,
                          label in enumerate(unique_labels)}

        encoded_ground_truth = np.array([label_to_index[label]
                                         for label in ground_truth])

        encoded_predictions = np.array([label_to_index[label]
                                        for label in predictions])

        num_classes = len(unique_labels)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(len(encoded_ground_truth)):
            confusion_matrix[encoded_ground_truth[i],
                             encoded_predictions[i]] += 1

        p_o = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        total_per_class = np.sum(confusion_matrix, axis=1)
        predicted_per_class = np.sum(confusion_matrix, axis=0)

        numerator = np.sum(total_per_class * predicted_per_class)
        denominator = np.sum(confusion_matrix) ** 2
        p_e = numerator / denominator

        kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 1.0
        return kappa


class MacroAverage(Metric):
    """
    A classification metric that returns the macro-average of the model,
    based on the predictions and on the ground truth.
    """
    def __call__(self, ground_truth: np.array, predictions: np.array) -> float:
        """
        The customised __call__ method for this metric, based on the blueprint
        in Metric class.
        """
        classes = np.unique(ground_truth)
        precision_per_class = []
        for c in classes:
            true_positive = np.sum((ground_truth == c) & (predictions == c))
            predicted_positive = np.sum(predictions == c)
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
    def __call__(self, ground_truth: np.array, predictions: np.array) -> float:
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
