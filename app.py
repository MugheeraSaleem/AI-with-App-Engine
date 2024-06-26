# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:00:06 2024

@author: mughe
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__),
                                                'linear_model.h5'))

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get the x value from the request
        x_value = float(request.args.get('x'))
        
        # Reshape the input for the model
        x_input = np.array([[x_value]])
        
        # Predict using the model
        prediction = model.predict(x_input)
        
        # Return the prediction as a JSON response
        return jsonify({'y_intercept': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
