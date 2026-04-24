from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Load the trained XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Define the exact feature names expected by the model
        feature_names = [
            'gender', 'age', 'study_hours_per_week', 'attendance_rate', 
            'parent_education', 'internet_access', 'extracurricular', 
            'previous_score', 'final_score'
        ]
        
        # Extract features in the correct order to form a single row DataFrame
        input_dict = {feature: [data.get(feature, 0)] for feature in feature_names}
        input_df = pd.DataFrame(input_dict)
        
        # Make the prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1] # Probability of the positive class
        
        # Return the results
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    # Run the app on 0.0.0.0 to make it accessible externally (required for AWS)
    app.run(host='0.0.0.0', port=8080)
