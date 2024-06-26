# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:47:00 2024

@author: mughe
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os

# Load the data
x = np.load(os.path.join(os.path.dirname(__file__),'Data','x.npy'))
y = np.load(os.path.join(os.path.dirname(__file__),'Data','y.npy'))

# Define the neural network model
model = Sequential([
    Dense(2, activation='relu', input_shape=(1,)),
    Dense(4, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x, y, epochs=10)

# make a sample request to see the desired results are being achieved
req = 10
print('model prediction:',model.predict(np.array([[float(req)]])))
print('Exact solution:',2.5*req+1.0)

# Save the model
model.save(os.path.join(os.path.dirname(__file__),'linear_model.h5'))
