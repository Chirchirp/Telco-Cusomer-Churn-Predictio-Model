import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the saved Random Forest model
model_path = os.path.join('model', 'random_forest_model.pkl')

# Check if the model file exists before loading
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

rf_model = joblib.load(model_path)

# Define the label encoders for categorical variables
label_encoders = {
    'Offer': LabelEncoder(),
    'Internet Service': LabelEncoder(),
    'Internet Type': LabelEncoder(),
    'Payment Method': LabelEncoder(),
    'Gender': LabelEncoder()
}

# Fit label encoders with actual categories
label_encoders['Offer'].fit(['None', 'Discount', 'Bundle', 'Loyalty'])
label_encoders['Internet Service'].fit(['DSL', 'Fiber optic', 'No'])
label_encoders['Internet Type'].fit(['Cable', 'Fiber', 'DSL'])
label_encoders['Payment Method'].fit(['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'])
label_encoders['Gender'].fit(['Male', 'Female'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from the form
        input_features = np.array([[ 
            int(request.form['referred_friend']),
            int(request.form['number_of_referrals']),
            int(request.form['tenure']),
            label_encoders['Offer'].transform([request.form['offer']])[0],
            int(request.form['phone_service']),
            float(request.form['avg_monthly_ld_charges']),
            int(request.form['multiple_lines']),
            label_encoders['Internet Service'].transform([request.form['internet_service']])[0],
            label_encoders['Internet Type'].transform([request.form['internet_type']])[0],
            float(request.form['avg_monthly_gb_download']),
            int(request.form['contract']),
            label_encoders['Payment Method'].transform([request.form['payment_method']])[0],
            float(request.form['monthly_charge']),
            float(request.form['total_charges']),
            float(request.form['total_refunds']),
            float(request.form['total_extra_data_charges']),
            float(request.form['total_ld_charges']),
            float(request.form['total_revenue']),
            label_encoders['Gender'].transform([request.form['gender']])[0],
            int(request.form['age']),
            int(request.form['under_30']),
            int(request.form['senior_citizen']),
            int(request.form['married']),
            float(request.form['churn_score']),
            int(request.form['contract_period']),
            float(request.form['contract_tenure_ratio'])
        ]])

        # Make the prediction
        prediction = rf_model.predict(input_features)
        prediction_text = "The customer is likely to churn." if prediction[0] == 1 else "The customer is not likely to churn."

    except Exception as e:
        prediction_text = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
