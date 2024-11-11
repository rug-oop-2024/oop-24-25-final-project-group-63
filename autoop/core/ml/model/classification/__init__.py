from ..classification.k_nearest_neighbors import KNearestNeighbors
from ..classification.logistic_regression import LogisticRegression
from ..classification.random_forest_classifier import RandomForestClassifier

CLASSIFICATION_MODELS = {
    "k_nearest_neighbors": KNearestNeighbors(),
    "logistic_regression": LogisticRegression(),
    "random_forest_classifier": RandomForestClassifier()
}
