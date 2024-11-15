import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import get_model

st.set_page_config(page_title="Pipeline Deployment")

st.write("# Pipeline Deployment")
st.write("View saved pipelines, load them, and use them for predictions.")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

if not pipelines:
    st.write("No saved pipelines found.")
else:
    pipeline_names = [p.name + " Version: " + p.version for p in pipelines]
    pipe_name = st.selectbox("Select a Pipeline:", pipeline_names)
    selec_pipeline = next(
        p for p in pipelines if p.name + " Version: " + p.version == pipe_name)

    st.write("## Pipeline Summary")
    if selec_pipeline:
        pipeline_data = pickle.loads(selec_pipeline.data)

        st.write(f"**Pipeline Name:** {selec_pipeline.name}")
        st.write(f"**Pipeline Version:** {selec_pipeline.version}")
        st.write(f"**Model Type:** {pipeline_data['model']}")
        st.write("**Input Features:**", ", ".join(
            [feat.name for feat in pipeline_data['input_features']]))
        st.write("**Target Feature:**", pipeline_data['target_feature'].name)
        st.write("**Metrics:**", [str(metric)
                                  for metric in pipeline_data['metrics']])

        if st.button("Delete Pipeline"):
            automl.registry.delete(selec_pipeline.id)
            st.success(f"Pipeline '{selec_pipeline.name}' has been deleted.")
            st.rerun()

        st.write("## Predict with Pipeline")
        uploaded_file = st.file_uploader("Upload a CSV file for prediction",
                                         type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)

            dataset = Dataset.from_dataframe(
                name="prediction_dataset",
                asset_path=None,
                data=data
            )

            pipeline = Pipeline(
                metrics=pipeline_data['metrics'],
                dataset=dataset,
                model=get_model(pipeline_data['model']),
                input_features=pipeline_data['input_features'],
                target_feature=pipeline_data['target_feature'],
                split=pipeline_data['split']
            )

            result = pipeline.execute()

            st.write("## Predictions")
            st.write(result['predictions'])
