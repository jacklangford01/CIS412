import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
import numpy as np

# (Optional) helps unpickling but not strictly required if sklearn is installed
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Airline Passenger Satisfaction Prediction", page_icon="✈️")

# ---------- Paths ----------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "decision_tree_model.pkl"   # your pruned tree
DATA_INFO_PATH = HERE / "data_info.pkl"         # must contain expected_columns, etc.

# ---------- Load artifacts ----------
@st.cache_resource
def load_pickle(p: Path):
    with p.open("rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}.\n{e}")
    st.stop()

try:
    data_info = load_pickle(DATA_INFO_PATH)
except Exception as e:
    st.error(
        f"Could not load data_info at {DATA_INFO_PATH}.\n"
        f"Ensure data_info.pkl exists and includes expected_columns.\n{e}"
    )
    st.stop()

expected_columns = data_info["expected_columns"]
numeric_ranges = data_info["numeric_ranges"]
categorical_unique_values = data_info["categorical_unique_values"]
feature_order = data_info["feature_order"]
ohe_categorical_columns = data_info["ohe_categorical_columns"]
numeric_columns = data_info["numeric_columns"]

# Helper: label->code for UI selections (not needed if we directly use values)
# def label_to_code(selection_label: str, mapping: dict) -> str:
#     inv = {v: k for k, v in mapping.items()}
#     return inv[selection_label]

# ---------- UI ----------
st.title("Airline Passenger Satisfaction Prediction")
st.caption("Predicting passenger satisfaction using a Decision Tree Classifier.")

st.header("Enter Passenger Details")

def num_slider(name, default_val=None): # Removed default, lo, hi, step; now relies on data_info
    r = numeric_ranges.get(name, {})
    lo = int(r.get("min"))
    hi = int(r.get("max"))
    val = int(r.get("default")) if default_val is None else default_val
    
    # Handle specific features where slider step might not be 1
    if name == 'Departure Delay in Minutes' or name == 'Arrival Delay in Minutes':
        step = 1 # or maybe smaller step for granular control
    elif name in ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                  'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
                  'Inflight entertainment', 'On-board service', 'Leg room service', 
                  'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']:
        step = 1
    else:
        step = 1 # Default step

    return st.slider(name.replace("_", " ").title(), min_value=lo, max_value=hi, value=val, step=step)

# Store user inputs
user_inputs = {}

st.subheader("Numerical Features")
for col in numeric_columns:
    user_inputs[col] = num_slider(col)

st.subheader("Categorical Features")
for col in ohe_categorical_columns:
    options = categorical_unique_values.get(col, [])
    if options:
        user_inputs[col] = st.selectbox(col.replace("_", " ").title(), options)

# ---------- Build raw row ----------
raw_row = {k: user_inputs[k] for k in feature_order}
raw_df = pd.DataFrame([raw_row])

# ---------- Encode EXACTLY like training ----------
# OHE only these categorical code columns; drop_first=True
input_encoded = pd.get_dummies(raw_df, columns=ohe_categorical_columns, drop_first=True, dtype=int)

# Make sure all expected training columns exist and in the same order
for col in expected_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_columns]

st.divider()
if st.button("Predict Satisfaction"):
    try:
        pred = model.predict(input_encoded)
        proba = getattr(model, "predict_proba", None)

        st.subheader("Prediction Result")
        if pred[0] == 0: # 0 for 'neutral or dissatisfied'
            st.error("Prediction: Neutral or Dissatisfied")
        else: # 1 for 'satisfied'
            st.success("Prediction: Satisfied")

        if callable(proba):
            p = proba(input_encoded)[0]
            st.write(f"Probability of Neutral or Dissatisfied: {p[0]:.2f}")
            st.write(f"Probability of Satisfied: {p[1]:.2f}")
    except Exception as e:
        st.error(f"Inference failed: {e}")
