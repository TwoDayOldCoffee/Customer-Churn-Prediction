import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_prediction_model.pkl")
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Telecom Customer Churn Prediction")
st.write("Adjust customer attributes and predict churn probability.")

st.sidebar.header("Customer Information")
AccountWeeks = st.sidebar.slider("Account Weeks", 0, 200, 50)
ContractRenewal = st.sidebar.checkbox("Contract Renewal")
DataPlan = st.sidebar.checkbox("Has Data Plan")
DataUsage = st.sidebar.slider("Monthly Data Usage (GB)", 0.0, 20.0, 5.0)
CustServCalls = st.sidebar.slider("Customer Service Calls", 0, 10, 1)
DayMins = st.sidebar.slider("Day Minutes", 0.0, 400.0, 180.0)
DayCalls = st.sidebar.slider("Day Calls", 0, 200, 100)
MonthlyCharge = st.sidebar.slider("Monthly Charge ($)", 0.0, 200.0, 70.0)
OverageFee = st.sidebar.slider("Overage Fee ($)", 0.0, 50.0, 5.0)
RoamMins = st.sidebar.slider("Roaming Minutes", 0.0, 50.0, 2.0)

usage_per_week = DayMins/7
calls_per_week = DayCalls/7
revenue_per_week = MonthlyCharge/4
high_data_usage = int(DataUsage>5)
high_service_calls = int(CustServCalls>3)

input_data = pd.DataFrame({
    'AccountWeeks':[AccountWeeks],
    'ContractRenewal':[int(ContractRenewal)],
    'DataPlan':[int(DataPlan)],
    'DataUsage':[DataUsage],
    'CustServCalls':[CustServCalls],
    'DayMins':[DayMins],
    'DayCalls':[DayCalls],
    'MonthlyCharge':[MonthlyCharge],
    'OverageFee':[OverageFee],
    'RoamMins':[RoamMins],
    'usage_per_week':[usage_per_week],
    'calls_per_week':[calls_per_week],
    'revenue_per_week':[revenue_per_week],
    'high_data_usage':[high_data_usage],
    'high_service_calls':[high_service_calls]
})

st.subheader("Derived Features")
st.write("Usage per Week:", round(usage_per_week,2))
st.write("Calls per Week:", round(calls_per_week,2))
st.write("Revenue per Week:", round(revenue_per_week,2))

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Customer likely to churn ({probability:.2%} probability)")
    else:
        st.success(f"Customer likely to stay ({1-probability:.2%} confidence)")
    st.subheader("Churn Probability")
    st.progress(int(probability * 100))
    st.write("Model confidence:", round(probability,3))
