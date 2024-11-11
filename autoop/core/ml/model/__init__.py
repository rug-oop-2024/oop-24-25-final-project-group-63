from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import REGRESSION_MODELS
from autoop.core.ml.model.classification import CLASSIFICATION_MODELS


def get_model(model_name: str) -> Model:
    """
        Factory function to get a model by name.

        Args:
            model_name: the string of the name of a model from the lists.
        """
    if model_name in CLASSIFICATION_MODELS:
        return CLASSIFICATION_MODELS[model_name]
    elif model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]
    else:
        raise TypeError("This model is not implemented!")
