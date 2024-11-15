from ..regression.multiple_linear_regression import MultipleLinearRegression
from ..regression.gradient_boost_regressor import GradientBoostModel
from ..regression.support_vector_regression import SVRModel

REGRESSION_MODELS = {
    "multiple linear regression": MultipleLinearRegression(),
    "gradient boost": GradientBoostModel(),
    "support vector regression": SVRModel()
}
