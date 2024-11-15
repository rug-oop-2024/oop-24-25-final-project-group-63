from ..classification.k_nearest_neighbors import KNearestNeighbors
from ..classification.decision_tree_classifier import DecisionTreeModel
from ..classification.random_forest_class import RandomForestClassifierModel

CLASSIFICATION_MODELS = {
    "k nearest neighbors": KNearestNeighbors(),
    "decision tree classifier": DecisionTreeModel(),
    "random forest classifier": RandomForestClassifierModel()
}
