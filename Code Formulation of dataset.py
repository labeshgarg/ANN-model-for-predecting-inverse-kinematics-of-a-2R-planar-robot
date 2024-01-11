
''' Formulation of Data Set with input X(Theta1 and Theta2)
and corresponding Output Y(X_Cord and Y_Coord of end-effector)'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DH Parameters, only link length 'a'
a = np.array([50, 20])

# Initialize variables
numPoints = 1
endEffectorPositions = []

# Loop through joint angles
for theta1_deg in np.arange(0, 360, 2):
    for theta2_deg in np.arange(-136, 136, 2):
        # Convert degrees to radians
        theta1_rad = np.radians(theta1_deg)
        theta2_rad = np.radians(theta2_deg)

        # Calculate end-effector position
        X = a[0] * np.cos(theta1_rad) + a[1] * np.cos(theta1_rad + theta2_rad)
        Y = a[0] * np.sin(theta1_rad) + a[1] * np.sin(theta1_rad + theta2_rad)

        # Store the end-effector position
        endEffectorPositions.append([X, Y, theta1_deg, theta2_deg])

# Convert the list to a NumPy array
endEffectorPositions = np.array(endEffectorPositions)

# Create a DataFrame from the endEffectorPositions array
df = pd.DataFrame(endEffectorPositions, columns=['X', 'Y', 'Theta1_deg', 'Theta2_deg'])

# Define the file path for saving the CSV
output_file_path = '/mydataset.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)
