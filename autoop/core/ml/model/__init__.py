from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import REGRESSION_MODELS
from autoop.core.ml.model.classification import CLASSIFICATION_MODELS


def get_model(model_name: str) -> Model:
    """
        Factory function to get a model by name.

        Args:
            model_name: the string of the name of a model from the lists.
        """
    # match model_name:
    #     case "multiple_linear_regression":
    #         return MultipleLinearRegression()
    #     case "lasso_regression":
    #         return Lasso()
    #     case "support_vector_regression":
    #         return SVR()
    #     case "k_nearest_neighbors":
    #         return KNearestNeighbors()
    #     case "logistic_regression":
    #         return LogisticRegression()
    #     case "random_forest_classifier":
    #         return RandomForestClassifier()
    #     case _:
    #         raise TypeError("This model is not implemented!")
    if model_name in CLASSIFICATION_MODELS:
        return CLASSIFICATION_MODELS[model_name]
    elif model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]
    else:
        raise TypeError("This model is not implemented!")
