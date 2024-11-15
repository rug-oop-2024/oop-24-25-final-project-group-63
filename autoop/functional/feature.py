from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Assumption: only categorical and numerical features and no NaN values.
    "threshold" value is used when the data tries to be transformed into
    numerical. However, there can also be categorical data that uses
    numbers, so it checks whether there are more than 5 types. If there
    are five different numbers, the data is considered numerical.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    threshold = 5
    features = []
    dataset = dataset.read()
    for column in dataset:
        try:
            converted_col = pd.to_numeric(dataset[column])
            if pd.api.types.is_numeric_dtype(converted_col):
                feature_type = "numerical"
            else:
                unique_values = dataset[column].nunique()
                if unique_values < threshold:
                    feature_type = "categorical"
                else:
                    feature_type = "numerical"
        except Exception:
            feature_type = "categorical"
        features.append(Feature(name=column, type=feature_type))
    return features
