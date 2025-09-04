import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="SarojRauth/best_Tourism_model", filename="best_Tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Best Tourism Products - Prediction App")
st.write("""
This application predicts the likelihood of a customer opting for a tourism product based on its given parameters.
Please enter the details to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18.0, max_value=61.0, value=25.0, step=1)
TypeofContact = st.selectbox("Type_of_Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5.0, max_value=36.0, value=25.0, step=1)
Gender = st.selectbox("Gender", ["Male", "Female", "Fe male"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1.0, max_value=5.0, value=2.0, step=1)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital_Status", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1.0, max_value=22.0, value=2.0, step=1)
Passport = st.selectbox("Passport", ["0", "1"])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", ["1", "2", "3", "4", "5"])
OwnCar = st.selectbox("OwnCar", ["0", "1"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0.0, max_value=3.0, value=2.0, step=1)
Designation = st.selectbox("Designation", ["AVP", "Manager", "Executive", "Senior Manager","VP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000.0, max_value=98678.0, value=1000.0)




# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'Type_of_Contact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'Marital_Status': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict ProdTaken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "No Product Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
