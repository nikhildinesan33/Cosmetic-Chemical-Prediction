import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cosmetic Chemical Prediction System",
    page_icon="🧪",
    layout="centered"
)

# Load trained objects
model = joblib.load("models/chemical_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

lab_product = joblib.load("models/product_lab.pkl")
lab_company = joblib.load("models/company_lab.pkl")
lab_brand = joblib.load("models/brand_lab.pkl")
lab_primary = joblib.load("models/primary_lab.pkl")
lab_sub = joblib.load("models/sub_lab.pkl")
lab_chemical = joblib.load("models/label_encoder.pkl")


# UI
st.title("Cosmetic Chemical Prediction System")
st.markdown(
    "Select the product details below to predict the chemical used "
    "based on the trained machine learning model."
)

st.sidebar.header("Input parameters")
st.sidebar.markdown("Please select all required fields")

# Dropdown Inputs (SAFE)
ProductName = st.sidebar.selectbox("Product Name", lab_product.classes_)
CompanyName = st.sidebar.selectbox("Company Name", lab_company.classes_)
BrandName = st.sidebar.selectbox("Brand Name", lab_brand.classes_)
PrimaryCategory = st.sidebar.selectbox("Primary Category", lab_primary.classes_)
SubCategory = st.sidebar.selectbox("Sub Category", lab_sub.classes_)
ChemicalCount = st.sidebar.number_input("Chemical Count", min_value=0, value=1)

# Encode Inputs
input_df = pd.DataFrame({
    "ProductName": [lab_product.transform([ProductName])[0]],
    "CompanyName": [lab_company.transform([CompanyName])[0]],
    "BrandName": [lab_brand.transform([BrandName])[0]],
    "PrimaryCategory": [lab_primary.transform([PrimaryCategory])[0]],
    "SubCategory": [lab_sub.transform([SubCategory])[0]],
    "ChemicalCount": [ChemicalCount]
})

# Prediction
if st.button("Predict Chemical"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    chemical_name = lab_chemical.inverse_transform(prediction)

    st.success(f"Predicted Chemical: **{chemical_name[0]}**")
