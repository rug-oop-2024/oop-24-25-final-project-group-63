from ..classification.k_nearest_neighbors import KNearestNeighbors
from ..classification.logistic_regression import LogisticRegressionModel
from ..classification.random_forest_classifier import RandomForestClassifierModel

CLASSIFICATION_MODELS = {
    "k_nearest_neighbors": KNearestNeighbors(),
    "logistic_regression": LogisticRegressionModel(),
    "random_forest_classifier": RandomForestClassifierModel()
}
