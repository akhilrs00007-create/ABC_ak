
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Ensure 'delivery_delay.sav' is in the same directory as this script when running Streamlit
try:
    model = joblib.load('delivery_delay.sav')
except FileNotFoundError:
    st.error("Error: 'delivery_delay.sav' model file not found. Please ensure it's in the same directory.")
    st.stop()

# Define the feature names from X.columns (as determined earlier in the notebook)
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot',
    'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score',
    'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'
]

st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature in ['Delivery_Distance', 'Package_Weight', 'Fuel_Efficiency']:
        input_data[feature] = st.number_input(f'{feature}:', value=10.0, format="%.2f")
    elif feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Num_Stops', 'Road_Condition_Score', 'Warehouse_Processing_Time']:
        input_data[feature] = st.slider(f'{feature}:', min_value=1, max_value=10, value=5)
    elif feature in ['Driver_Experience', 'Vehicle_Age']:
        input_data[feature] = st.slider(f'{feature}:', min_value=0, max_value=20, value=5)


if st.button('Predict Delivery Delay'):
    # Convert input data to a Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error('The model predicts a **Delivery Delay**.')
    else:
        st.success('The model predicts **No Delivery Delay**.')

    st.write(f"Probability of No Delay: {prediction_proba[0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[1]:.2f}")

