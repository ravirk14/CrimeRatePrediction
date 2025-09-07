import os
from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import joblib
import numpy as np
from geopy.geocoders import Nominatim

app = Flask(__name__)

# A dummy path for the images folder, adjust if necessary
app.config['images'] = os.path.join('static', 'images')

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/images/<filename>')
def download_file(filename):
    return send_from_directory(app.config['images'], filename)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/work.html')
def work():
    return render_template('work.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/result.html', methods=['POST'])
def predict():
    # Load the trained model
    rfc = joblib.load('rf_model')
    print('Model loaded')

    if request.method == 'POST':
        # 1. Get location and timestamp from the form
        address = request.form['Location']
        dt_string = request.form['timestamp']

        # 2. Convert location name to latitude and longitude
        geolocator = Nominatim(user_agent="crime_predictor_app_final_v2") # Use a unique agent
        location = geolocator.geocode(address, timeout=None)
        
        if location is None:
            return render_template('result.html', prediction="Error: Location could not be found. Please try again with a more specific address.")

        lat = location.latitude
        log = location.longitude
        print(f"Location found: {location.address} -> Lat: {lat}, Lon: {log}")

        # 3. Convert string to a single pandas Timestamp object
        timestamp = pd.to_datetime(dt_string, dayfirst=True)

        # 4. **FINAL ROBUST METHOD: Create a simple list of features**
        # To bypass the strange issue with .weekday, we will use .dayofweek for both.
        
        working_dayofweek = timestamp.dayofweek # This is a number (e.g., 5 for Saturday)

        features_list = [
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.dayofyear,
            timestamp.isocalendar().week,
            timestamp.isocalendar().week,  # for weekofyear
            working_dayofweek,            # for dayofweek
            working_dayofweek,            # for weekday (using the one we know works)
            timestamp.quarter,
            0,  # act13
            0,  # act279
            0,  # act323
            0,  # act363
            0,  # act302
            lat,
            log
        ]
        
        # 5. Convert list to a 2D NumPy array for the model
        final_features = np.array(features_list).reshape(1, -1)
        print(f"Features created for prediction: {final_features}")

        # 6. Make the prediction
        my_prediction = rfc.predict(final_features)

        # 7. Interpret the result
        if my_prediction[0] == 1:
            prediction_text = 'Predicted crime: Act 379 - Robbery'
        else:
            prediction_text = 'Place is safe. No crime (Act 379) is expected at that timestamp.'

    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)