import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and encoder
model = joblib.load("recall_logreg.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("encoder.pkl")

def preprocess_data(df):
    # Drop irrelevant columns
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, handle non-numeric values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop target column if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    
    # Define categorical columns needed for selected features
    categorical_cols = ['Contract', 'InternetService']
    numerical_cols = ['tenure', 'MonthlyCharges']
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Expected columns from training
    expected_columns = [
        'tenure', 'MonthlyCharges',
        'Contract_One year', 'Contract_Two year',
        'InternetService_Fiber optic', 'InternetService_No'
    ]
    
    # Add missing columns with 0s
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Subset and reorder to selected features
    df_encoded = df_encoded[expected_columns]
    
    # Scale features
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

# Streamlit app
st.title("Telco Customer Churn Prediction (Minimal Features)")

# Option 1: Upload CSV for batch predictions
st.header("Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    try:
        processed_df = preprocess_data(df)
        y_pred = model.predict(processed_df)
        y_pred_labels = le.inverse_transform(y_pred)
        df['Predicted_Churn'] = y_pred_labels
        st.write("Predictions:")
        st.dataframe(df[['customerID', 'Predicted_Churn']] if 'customerID' in df.columns else df[['Predicted_Churn']])
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")

# Option 2: Manual input for single prediction
st.header("Enter Customer Details")
with st.form("churn_form"):
    # Numerical inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    
    # Categorical inputs
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        # Create DataFrame from inputs
        df = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'Contract': [contract],
            'InternetService': [internet_service]
        })
        
        try:
            processed_df = preprocess_data(df)
            y_pred = model.predict(processed_df)
            y_pred_label = le.inverse_transform(y_pred)[0]
            st.success(f"Predicted Churn: **{y_pred_label}**")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")