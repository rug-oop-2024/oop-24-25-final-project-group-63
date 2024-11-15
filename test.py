# from sklearn.datasets import fetch_openml
# from autoop.functional.feature import detect_feature_types
# # from autoop.core.ml.feature import Feature
# import pandas as pd
# from autoop.core.ml.dataset import Dataset
# import io
import pandas as pd
import numpy as np
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.regression import MultipleLinearRegression, RandomForestRegressor
from autoop.core.ml.metric import get_metric


def run_models():
    # data_knn = pd.read_csv(
    #     "data/student_performance.csv"
    # )
    # observations_knn = np.column_stack(
    #     (data_knn["AttendanceRate"], data_knn["StudyHoursPerWeek"],
    #      data_knn["ParentalSupport"])
    # )
    # ground_truth_knn = data_knn["FinalGrade"]

    data_continuous = pd.read_csv(
        "data/linear_regression_data.csv"
    )
    observations_cont = np.column_stack((data_continuous["age"],
                                        data_continuous["experience"]))
    ground_truth_cont = data_continuous["income"]

    # k_nearest_neighbors
    # model_knn = KNearestNeighbors(k=3)
    # model_knn.fit(observations_knn, ground_truth_knn)

    # data_obs = pd.read_csv(
    #     "data/student_performance.csv"
    # )

    # separate_obs = np.column_stack(
    #     (data_obs["AttendanceRate"], data_obs["StudyHoursPerWeek"],
    #      data_obs["ParentalSupport"])
    # )

    # predictions_KNN = model_knn.predict(separate_obs)
    # print("KNN", predictions_KNN)

    # class_metric = get_metric("cohens_kappa")
    # print("metric measurement:", class_metric(ground_truth_knn,
    # predictions_KNN))
    # model_log = LogisticRegression()
    # model_log.fit(observations_knn, ground_truth_knn)

    # predictions_log = model_log.predict(observations_knn)
    # print("logistic regression", predictions_log)

    # model_forest = RandomForestClassifier()
    # model_forest.fit(observations_knn, ground_truth_knn)

    # predictions_forest = model_forest.predict(observations_knn)
    # print("logistic regression", predictions_forest)

    # # multiple linear regression
    # model_linear = MultipleLinearRegression()

    # model_linear.fit(observations_cont, ground_truth_cont)
    # pred_linear = model_linear.predict(observations_cont)

    # regres_metric = get_metric("R_squared")
    # print(regres_metric(ground_truth_cont, pred_linear))

    # # sklearn_wrap
    model_lasso = RandomForestRegressor()
    model_lasso.fit(observations_cont, ground_truth_cont)
    pred_lasso = model_lasso.predict(observations_cont)
    print("LASSO", pred_lasso)
    print("CORRECT", [element for element in ground_truth_cont])

    # print(regres_metric(ground_truth_cont, pred_lasso))

    # model_svr = SVR()
    # model_svr.fit(observations_cont, ground_truth_cont)
    # pred_svr = model_svr.predict(observations_cont)

    # print(regres_metric(ground_truth_cont, pred_svr))


if __name__ == "__main__":
    run_models()
