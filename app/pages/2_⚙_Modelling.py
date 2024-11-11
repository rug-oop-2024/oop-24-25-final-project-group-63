import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
# import below are just imported for testing purposes
import numpy as np
import time


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
st.write("This is a test for the slider, we might implement it for the percentage used for fitting.")
x = st.slider('x')
st.write(x, 'Value of squared is', x * x)

st.write("This is a test")
st.text_input("Your name", key="name")
st.session_state.name

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
    df['first column'])

'You selected: ', option

add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)


left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


latest_iteration = st.empty()
bar = st.progress(0)

for i in range(1):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    # time.sleep(0.01)

'...and now we\'re done!'

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

st.button("Click me")
# st.download_button("Download file", data)
# st.link_button("Go to gallery", url)
# st.page_link("app.py", label="Datasets")
# st.data_editor("Edit data", data)
st.checkbox("I agree")
st.feedback("thumbs")
# st.pills("Tags", ["Sports", "Politics"])
st.radio("Pick one", ["cats", "dogs"])
# st.segmented_control("Filter", ["Open", "Closed"])
st.toggle("Enable")
st.selectbox("Pick one", ["cats", "dogs"])
st.multiselect("Buy", ["milk", "apples", "potatoes"])
st.slider("Pick a number", 0, 100)
st.select_slider("Pick a size", ["S", "M", "L"])
st.text_input("First name")
st.number_input("Pick a number", 0, 10)
st.text_area("Text to translate")
st.date_input("Your birthday")
st.time_input("Meeting time")
st.file_uploader("Upload a CSV")
# st.audio_input("Record a voice message")
st.camera_input("Take a picture")
st.color_picker("Pick a color")

# Use widgets' returned values in variables:
# for i in range(int(st.number_input("Num:"))):
#     foo()
# if st.sidebar.selectbox("I:",["f"]) == "f":
#     b()
# my_slider_val = st.slider("Quinn Mallory", 1, 88)
# st.write(slider_val)

# Disable widgets to remove interactivity:
# st.slider("Pick a number", 0, 100, disabled=True)