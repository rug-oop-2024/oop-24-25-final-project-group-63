from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    A class for constructing and executing a machine learning pipeline.

    The pipeline handles preprocessing, splitting data, training a model,
    and evaluating its performance using specified metrics.

    Attributes:
        metrics (List[Metric]): A list of metrics for evaluation.
        dataset (Dataset): The dataset to be used.
        model (Model): The machine learning model.
        input_features (List[Feature]): Features used as inputs to the model.
        target_feature (Feature): The target feature for prediction.
        split (float): Ratio of the dataset used for training (default: 0.8).
        artifacts (dict): Artifacts generated during pipeline execution.
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """
        Initializes the Pipeline class.

        Args:
            metrics (List[Metric]): Metrics used for evaluation.
            dataset (Dataset): The dataset for training and testing.
            model (Model): The model to be trained.
            input_features (List[Feature]): Features to be used as input.
            target_feature (Feature): The target variable.
            split (float): Ratio for training/testing split (default: 0.8).
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

    def __str__(self):
        """
        Provides a string representation of the pipeline.

        Returns:
            str: A detailed description of the pipeline components.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self):
        """
        Returns the model used in the pipeline.

        Returns:
            Model: The machine learning model.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Collects artifacts generated during the pipeline execution.

        Returns:
            List[Artifact]: A list of artifacts to be saved.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        mo = self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        artifacts.append(mo)
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """
        Registers an artifact during the pipeline process.

        Args:
            name (str): Name of the artifact.
            artifact: The artifact object.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """
        Preprocesses input and target features from the dataset.

        The features are preprocessed, and the artifacts are registered.
        """
        target_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_name, artifact)

        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [data for _, data, _ in input_results]

    def _split_data(self):
        """
        Splits the dataset into training and testing sets based on
        the split ratio.
        """
        split_idx = int(self._split * len(self._input_vectors[0]))
        self._train_X = [vector[:split_idx] for vector in self._input_vectors]
        self._test_X = [vector[split_idx:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split_idx]
        self._test_y = self._output_vector[split_idx:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compacts a list of feature vectors into a single 2D array.

        Args:
            vectors (List[np.array]): List of feature vectors.

        Returns:
            np.array: A combined 2D array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self):
        """
        Evaluates the model on the test set using specified metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self):
        """
        Executes the pipeline by preprocessing data, splitting it, training
        the model, and evaluating its performance.

        Returns:
            dict: A dictionary containing metrics results and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
