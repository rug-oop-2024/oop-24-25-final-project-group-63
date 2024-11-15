from ..regression.multiple_linear_regression import MultipleLinearRegression
from ..regression.gradient_boost_regressor import GradientBoostModel
from ..regression.support_vector_regression import SVRModel

REGRESSION_MODELS = {
    "multiple_linear_regression": MultipleLinearRegression(),
    "lasso_regression": GradientBoostModel(),
    "support_vector_regression": SVRModel()
}
