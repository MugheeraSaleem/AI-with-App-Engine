# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:42:59 2024

@author: mughe
"""

import numpy as np
import os

# Parameters for the straight line equation
m = 2.5  # Slope
c = 1.0  # Intercept

# Generate x values
x = np.linspace(0, 1000, 5000)
print(len(x))
# Generate y values using the straight line equation
y = m * x + c

# Save the data to .npy files
np.save(os.path.join(os.path.dirname(__file__),'Data','x.npy'),x)
np.save(os.path.join(os.path.dirname(__file__),'Data','y.npy'),y)