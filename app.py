from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pickle model
model = pickle.load(open('RF_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    answers = data.get('answers', [])
    
    if len(answers) != 25:
        return jsonify({'error': 'Invalid input. Ensure 25 answers are provided.'}), 400
    
    try:
        # Predict using the model
        prediction = model.predict([answers])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
