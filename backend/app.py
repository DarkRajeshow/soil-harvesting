from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)


defaltCrops = [{'crop': 'Carrots', 'confidence': 0.0}, {'crop': 'Potatoes', 'confidence': 0.0}, {'crop': 'Onions', 'confidence': 0.0}, {'crop': 'Tomatoes', 'confidence': 0.0}, {'crop': 'Cotton', 'confidence':0.0}, {'crop': 'Sugarcane', 'confidence': 0.0}, {'crop': 'Soybeans', 'confidence': 0.0}, {'crop': 'Maize', 'confidence': 0.0}, {'crop': 'Wheat', 'confidence': 0.0}, {'crop': 'Rice', 'confidence': 0.0}]

# Load the pre-trained model
try:
    with open('soil_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Model file not found. Please ensure soil_model.pkl exists.")
    model = None

# Define crop names
CROPS = {
    0: "Rice", 1: "Wheat", 2: "Maize", 3: "Soybeans",
    4: "Cotton", 5: "Sugarcane", 6: "Tomatoes",
    7: "Potatoes", 8: "Carrots", 9: "Onions"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from request

        features = [
            float(data['ph']),
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium'])
        ]
        

        # Basic input validation
        if not (0 <= features[0] <= 14):
            return jsonify({"error": "pH must be between 0 and 14"}), 400
        if not all(0 <= x <= 100 for x in features[1:]):
            return jsonify({"error": "N, P, K values must be between 0 and 100"}), 400
        
        # Make prediction
        if model is not None:
            # Reshape features for model input
            features_array = np.array(features).reshape(1, -1)
            
            # Get probability scores for all crops
            probabilities = model.predict_proba(features_array)[0]
            
            # Get indices of top 3 crops
            top_3_indices = probabilities.argsort()[-10:][::-1]
            
            # Format recommendations
            recommendations = [
                {
                    "crop": CROPS[idx],
                    "confidence": round(float(probabilities[idx] * 100), 1)
                }
                for idx in top_3_indices
            ]
            
            print(recommendations)
            return jsonify({
                "success": True,
                "recommendations": recommendations
            })
        else:
            return jsonify({"error": "Model not loaded"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)