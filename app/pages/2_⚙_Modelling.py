import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model, REGRESSION_MODELS
from autoop.core.ml.model import CLASSIFICATION_MODELS


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

x = st.selectbox("Datasets:", [element.name for element in list_of_artifacts])
for element in list_of_artifacts:
    if element.name == x:
        x = element
        break

df = x.to_dataframe()
dataset = Dataset.from_dataframe(
            name=x.name,
            asset_path=x.asset_path,
            data=df,
        )

c = detect_feature_types(dataset)

selected_features = st.multiselect("Select input features:", df.columns)

rest_of_features = []

for element in c:
    if element.name not in selected_features:
        rest_of_features.append(element.name)

target_feature = st.radio("Pick the targeted feature:", rest_of_features)
# for feature in c:
#     if feature.name in selected_features:
#         st.write(str(feature))
for feature in c:
    if feature.name == target_feature:
        target_feature = feature
        st.write(str(feature))

if target_feature.type == "categorical":
    selected_model = st.radio("Pick one model:", CLASSIFICATION_MODELS)
elif target_feature.type == "numerical":
    selected_model = st.radio("Pick one model:", REGRESSION_MODELS)

model = get_model(selected_model)
