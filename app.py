from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    rf_model = joblib.load('rf_model.pkl')
    print("Model successfully loaded.")
except Exception as e:
    print(f"Error loading the model: {e}")

# Define a basic route
@app.route('/')
def home():
    return render_template('home.html')

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = [
            float(data['Depot']),
            float(data['BusNo']),
            float(data['BusType']),
            float(data['Shift']),
            float(data['Act_start_Month']),
            float(data['UniqueRouteID']),
            float(data['TotalKM']),
            float(data['Hour']),
            float(data['Minute']),
            float(data['Act_start_Day_of_Week']),
            float(data['Act_start_Season']),
            float(data['temp']),
            float(data['icon'])
        ]
        features = np.array(features).reshape(1, -1)
        prediction = rf_model.predict(features)
        return render_template('home.html', prediction_text=f'The Predicted Energy is {prediction[0]}')
    except Exception as e:
        return render_template('home.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)


