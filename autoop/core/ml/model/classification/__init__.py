from ..classification.k_nearest_neighbors import KNearestNeighbors
from ..classification.logistic_regression import LogisticRegressionModel
from ..classification.random_forest_classifier import RandomForestClassifierModel

CLASSIFICATION_MODELS = {
    "k nearest neighbors": KNearestNeighbors(),
    "logistic regression": LogisticRegressionModel(),
    "random forest classifier": RandomForestClassifierModel()
}
