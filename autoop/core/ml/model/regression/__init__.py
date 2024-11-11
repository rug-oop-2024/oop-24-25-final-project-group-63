from ..regression.multiple_linear_regression import MultipleLinearRegression
from ..regression.lasso_regression import Lasso
from ..regression.support_vector_regression import SVR

REGRESSION_MODELS = {
    "multiple_linear_regression": MultipleLinearRegression(),
    "lasso_regression": Lasso(),
    "support_vector_regression": SVR()
}
