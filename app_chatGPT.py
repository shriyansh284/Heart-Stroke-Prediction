import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# Load model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# --- Page Layout ---
st.set_page_config(page_title="💓 Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("💓 Heart Disease Risk Prediction")
st.markdown(
    "<h4 style='color:#ff4d4d;'>Check your heart health risk instantly</h4>",
    unsafe_allow_html=True
)
st.write("Fill in the details below to know your estimated risk level.")

# --- Input Sections ---
with st.expander("👤 Demographics"):
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])

with st.expander("🧪 Medical Measurements"):
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

with st.expander("📈 ECG & Stress Tests"):
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Prediction ---
if st.button("🔍 Predict"):
    with st.spinner("Analyzing your heart health..."):
        time.sleep(1.5)

        raw_input = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] * 100

        # --- Result Display ---
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease ({prob:.1f}%)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({100-prob:.1f}%)")

        # Gauge Chart for Probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Heart Disease Risk (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prediction==1 else "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
