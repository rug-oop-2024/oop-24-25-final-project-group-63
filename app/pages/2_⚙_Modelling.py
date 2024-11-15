import streamlit as st
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model, REGRESSION_MODELS
from autoop.core.ml.model import CLASSIFICATION_MODELS
from autoop.core.ml.metric import get_metric, REGRESSION_METRICS
from autoop.core.ml.metric import CLASSIFICATION_METRICS
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning " +
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

write_helper_text("Please choose the dataset you want to use" +
                  " for your modellling.")

list_of_artifacts = automl.registry.list()
datasets = automl.registry.list(type="dataset")
dataset = st.selectbox("Datasets:", [dataset.name for dataset in datasets])
for element in list_of_artifacts:
    if element.name == dataset:
        dataset = element
        break
if dataset:
    df = dataset.to_dataframe()
    dataset = Dataset.from_dataframe(
                name=dataset.name,
                asset_path=dataset.asset_path,
                data=df,
            )

    det_feat = detect_feature_types(dataset)

    selected_features = st.multiselect("Select input features:", df.columns)

    rest_of_features = []

    for element in det_feat:
        if element.name not in selected_features:
            rest_of_features.append(element.name)

    target_feature = st.radio("Pick the targeted feature:", rest_of_features)
    if target_feature:
        for feature in det_feat:
            if feature.name == target_feature:
                target_feature = feature
                st.write(str(feature))

        if target_feature.type == "categorical":
            selected_model = st.radio("Pick one model:", CLASSIFICATION_MODELS)
            metric_type = CLASSIFICATION_METRICS
        elif target_feature.type == "numerical":
            selected_model = st.radio("Pick one model:", REGRESSION_MODELS)
            metric_type = REGRESSION_METRICS

        input_features = [feature for feature in det_feat
                          if feature.name in selected_features]

        model = get_model(selected_model)
        selected_metric = st.selectbox("Choose a metric for model evaluation:",
                                       metric_type)

        metric = get_metric(selected_metric)
        st.subheader("Split Dataset")
        split_ratio = st.slider(
            "Training set ratio (remaining goes to validation):",
            min_value=0.1, max_value=0.9, value=0.8)

        st.subheader("Pipeline Summary")
        st.write(f"**Dataset**: {dataset.name}")
        st.write(f"**Model**: {selected_model}")
        st.write(f"**Input Features**: {', '.join(selected_features)}")
        st.write(f"**Target Feature**: {target_feature.name}")
        st.write(f"**Metric**: {selected_metric}")
        st.write(
            f"**Training Split**: {split_ratio * 100:.0f}% Training, " +
            f"{100 - split_ratio * 100:.0f}% Validation")

    if st.button("Train Model"):
        if target_feature and input_features:
            pipeline = Pipeline(
                metrics=[metric],
                dataset=dataset,
                model=model,
                input_features=input_features,
                target_feature=target_feature,
                split=split_ratio
            )

            result = pipeline.execute()

            st.subheader("Model Evaluation Results")

            left_column, right_column = st.columns(2)

            left_column.write("Prediction values:")

            right_column.write("Ground truth values:")

            left_column.table(result["predictions"])

            right_column.table(pipeline._test_y)

            for metric, value in result["metrics"]:
                st.markdown(f"- **{metric.__class__.__name__}**: {value:.4f}")

            st.success("Training and evaluation complete!")
        else:
            st.error("Please select at least one input and output feature.")

    st.write("## Save Pipeline")
    pipeline_name = st.text_input("Pipeline Name", "Pipeline1")
    pipeline_version = st.text_input("Pipeline Version", "1.0")

    if st.button("Save Pipeline"):
        if pipeline_name and pipeline_version and input_features:
            pipeline_data = {
                "input_features": input_features,
                "target_feature": target_feature,
                "split": split_ratio,
                "model": selected_model,
                "metrics": [metric]
            }
            serialized_pipeline = pickle.dumps(pipeline_data)
            asset_path = f"./pipelines/{pipeline_name}_{pipeline_version}.pkl"

            artifact = Artifact(
                name=pipeline_name,
                version=pipeline_version,
                data=serialized_pipeline,
                asset_path=asset_path,
                type="pipeline"
            )
            automl.registry.register(artifact)
            st.success(f"Pipeline '{pipeline_name}' version" +
                       f" '{pipeline_version}' saved successfully!")
        else:
            st.error("Please provide both a name, version, and input for" +
                     " the pipeline.")
