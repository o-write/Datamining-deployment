import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. LOAD MODEL DAN SCALER (.pkl)
with open('LR_model.pkl', 'rb') as file:
    LR_model = pickle.load(file)

with open('LR_scaler.pkl', 'rb') as file:
    LR_scaler = pickle.load(file)

with open('nb_model_smote.pkl', 'rb') as file:
    nb_model_smote = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_nb = pickle.load(file)


# 2. SETUP TAMPILAN & SIDEBAR NAVIGATION
st.sidebar.title("Dashboard Kesehatan")
st.sidebar.write("Pilih menu di bawah untuk melakukan analisis pasien:")
halaman = st.sidebar.radio(
    "Navigasi Halaman:",
    ["Prediksi Glucose (Linear Regression)", "Klasifikasi Outcome (Naive Bayes)"]
)

# 3. KONDISIONAL HALAMAN
if halaman == "Prediksi Glucose (Linear Regression)":
    st.title("📈 Prediksi Kadar Glucose Pasien")
    st.write("Masukkan data pasien untuk memprediksi kadar Glucose.")
    age = st.number_input("Age", min_value=0)
    preg = st.number_input("Pregnancies", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    bp = st.number_input("BloodPressure", min_value=0)
    hba1c = st.number_input("HbA1c", min_value=0.0)
    ldl = st.number_input("LDL", min_value=0.0)
    hdl = st.number_input("HDL", min_value=0.0)
    trig = st.number_input("Triglycerides", min_value=0.0)
    waist = st.number_input("WaistCircumference", min_value=0.0)
    hip = st.number_input("HipCircumference", min_value=0.0)
    whr = st.number_input("WHR", min_value=0.0)
    fam = st.selectbox("FamilyHistory", [0, 1])
    diet = st.selectbox("DietType", [0, 1])
    hyp = st.selectbox("Hypertension", [0, 1])
    med = st.selectbox("MedicationUse", [0, 1])

    if st.button("Prediksi Glucose"):
        input_data = pd.DataFrame([[age, preg, bmi, bp, hba1c, ldl, hdl, trig, waist, hip, whr, fam, diet, hyp, med]],
            columns=['Age', 'Pregnancies', 'BMI', 'BloodPressure', 'HbA1c', 'LDL', 'HDL',
                     'Triglycerides', 'WaistCircumference', 'HipCircumference', 'WHR',
                     'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse'])
        scaled_data = LR_scaler.transform(input_data)
        pred = LR_model.predict(scaled_data)
        st.success(f"Hasil Prediksi Kadar Glucose: {pred[0]:.2f}")

elif halaman == "Klasifikasi Outcome (Naive Bayes)":
    st.title("🩺 Klasifikasi Risiko Diabetes")
    st.write("Masukkan data pasien untuk diagnosis risiko diabetes.")
    age = st.number_input("Age", min_value=0)
    preg = st.number_input("Pregnancies", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    gluc = st.number_input("Glucose", min_value=0.0)
    bp = st.number_input("BloodPressure", min_value=0)
    hba1c = st.number_input("HbA1c", min_value=0.0)
    ldl = st.number_input("LDL", min_value=0.0)
    hdl = st.number_input("HDL", min_value=0.0)
    trig = st.number_input("Triglycerides", min_value=0.0)
    waist = st.number_input("WaistCircumference", min_value=0.0)
    hip = st.number_input("HipCircumference", min_value=0.0)
    whr = st.number_input("WHR", min_value=0.0)
    fam = st.selectbox("FamilyHistory", [0, 1])
    diet = st.selectbox("DietType", [0, 1])
    hyp = st.selectbox("Hypertension", [0, 1])
    med = st.selectbox("MedicationUse", [0, 1])

    if st.button("Klasifikasi Diabetes"):
        input_data = pd.DataFrame([[age, preg, bmi, gluc, bp, hba1c, ldl, hdl, trig, waist, hip, whr, fam, diet, hyp, med]],
            columns=['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 'LDL', 'HDL',
                     'Triglycerides', 'WaistCircumference', 'HipCircumference', 'WHR',
                     'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse'])

        scaled_data = scaler_nb.transform(input_data)
        pred = nb_model_smote.predict(scaled_data) 

        if pred[0] == 1:
            st.error("Hasil: Pasien Terindikasi Diabetes (Outcome = 1)")
        else:
            st.success("Hasil: Pasien Non-Diabetes (Outcome = 0)")
