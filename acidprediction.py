
import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Model and Preprocessors
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load('poly_model.pkl')
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly.pkl')
    return model, scaler, poly

model, scaler, poly = load_model()

# -----------------------------
# App Title and Description
# -----------------------------
st.title("Acid Concentration Prediction")

st.markdown("""
Predict **acid concentration (g/L)** using:
- **Temperature (°C)**
- **Conductivity (mS/cm)**

""")

# -----------------------------
# User Inputs
# -----------------------------
temperature = st.number_input(
    "Enter Temperature (°C):",
    min_value=0.0,
    value=15.0,
    step=1.0,
    format="%.1f"
)

conductivity_mS = st.number_input(
    "Enter Conductivity (mS/cm):",
    min_value=0.0,
    value=82.0,
    step=1.0,
    format="%.0f"
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Acid Concentration"):
    if temperature <= 0 or conductivity_mS <= 0:
        st.warning("Please enter valid temperature and conductivity values.")
    else:

        # Input must match training order
        X_input = np.array([[temperature, conductivity_mS]])

        # Apply same preprocessing as training
        X_scaled = scaler.transform(X_input)
        X_poly = poly.transform(X_scaled)

        prediction = model.predict(X_poly)

        st.success(
            f"Predicted Acid Concentration: **{prediction[0]:.2f} g/L**"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
**Model:** Polynomial Regression 
**Deployment:** Streamlit 
""")
