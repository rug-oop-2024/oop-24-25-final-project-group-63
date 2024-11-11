import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
st.header("Upload Dataset")
dataset_name = st.text_input("Dataset Name")
version = st.text_input("Version", "1.0.0")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if st.button("Upload Dataset"):
    if uploaded_file is not None and dataset_name:
        data = pd.read_csv(uploaded_file)
        dataset = Dataset.from_dataframe(
            data=data,
            name=dataset_name,
            asset_path=f"./datasets/{uploaded_file.name}",
            version=version
        )

        automl.registry.register(dataset)
        st.success(
            f"Dataset '{dataset_name}' uploaded and registered successfully")
        st.rerun()
    else:
        st.error("Provide both a dataset name and a file.")

st.header("Registered Datasets")

if datasets:
    for dataset in datasets:
        with st.expander(f"{dataset.name} (Version: {dataset.version})"):
            st.write(f"*Asset Path:* {dataset.asset_path}")
            st.write(f"*Tags:* {', '.join(dataset.tags) if dataset.tags else 'None'}")
            st.write(f"*Metadata:* {dataset.metadata if dataset.metadata else 'None'}")

            if st.button(f"View {dataset.name}", key=f"view-{dataset.name}"):
                df = dataset.read()
                st.write(df)

            if st.button(f"Delete {dataset.name}", key=f"delete-{dataset.name}"):
                automl.registry.delete(dataset.id)
                st.success(f"Dataset '{dataset.name}' deleted successfully.")
                st.rerun()
else:
    st.write("No datasets found.")
