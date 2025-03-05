import streamlit as st
import pandas as pd
import joblib
import os

# ✅ Function to Load Models
def load_models():
    """ Load trained models and encoders """
    model_dir = os.path.dirname(__file__)  # Get the directory where app.py is saved
    
    # ✅ Load models
    log_reg = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
    rf_regressor = joblib.load(os.path.join(model_dir, "random_forest_regressor.pkl"))
    
    # ✅ Load encoders & scaler
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    return log_reg, rf_regressor, label_encoders, scaler

# ✅ Function to Preprocess Data
def preprocess_data(input_data, label_encoders, scaler):
    """ Process input data: Encode categorical variables & scale regression features """

    # ✅ Check if required columns exist
    required_columns = ['book_table', 'votes', 'rate', 'listed_in(type)', 'online_order']
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None, None

    # ✅ Convert 'rate' column to numeric (Fixing '4.1/5' issue)
    input_data['rate'] = input_data['rate'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)

    # ✅ Convert 'online_order' to binary (if needed)
    if input_data['online_order'].dtype == object:
        input_data['online_order'] = input_data['online_order'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # ✅ Encode categorical variables
    for col in ['book_table', 'listed_in(type)']:
        if col in label_encoders:
            input_data[col] = input_data[col].astype(str)  # Ensure it's a string before encoding
            input_data[col] = label_encoders[col].transform(input_data[col])

    # ✅ Select Features
    X_classification = input_data[['book_table', 'votes', 'rate', 'listed_in(type)']]
    X_regression = input_data[['votes', 'rate', 'online_order', 'listed_in(type)']]

    # ✅ Fix: Convert 'rate' to numeric BEFORE scaling
    X_regression['rate'] = X_regression['rate'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)

    # ✅ Scale Regression Features
    X_regression_scaled = scaler.transform(X_regression)

    return X_classification, X_regression_scaled

# ✅ Main Function to Run Streamlit App
def main():
    st.title("🍽️ Zomato Restaurant Analysis & Prediction")
    st.write("Upload your dataset to predict **Online Order Availability & Estimated Cost for Two**.")

    # ✅ Load models
    log_reg, rf_regressor, label_encoders, scaler = load_models()

    # ✅ Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload Restaurant Data CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Uploaded Data Preview")
        st.dataframe(df.head())

        # ✅ Preprocess Data
        X_classification, X_regression_scaled = preprocess_data(df, label_encoders, scaler)

        if X_classification is not None and X_regression_scaled is not None:
            # ✅ Make Predictions
            predictions_cls = log_reg.predict(X_classification)
            predictions_reg = rf_regressor.predict(X_regression_scaled)

            # ✅ Add Predictions to DataFrame
            df['Predicted_Online_Order'] = predictions_cls
            df['Predicted_Cost'] = predictions_reg

            # ✅ Show Results
            st.write("### 🎯 Predictions")
            st.dataframe(df[['name', 'Predicted_Online_Order', 'Predicted_Cost']])

            # ✅ Download Predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
