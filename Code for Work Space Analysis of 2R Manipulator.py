# Work Space Analysis of 2R Manipulator
import numpy as np
import matplotlib.pyplot as plt

# DH Parameters, only link length 'a'

a = np.array([50, 20])
# Initialize variables
numPoints = 1
endEffectorPositions = []

# Loop through joint angles
for theta1 in np.arange(0, 2 * np.pi, np.pi / 20):
    for theta2 in np.arange(-3 * np.pi / 4, 3 * np.pi / 4, np.pi / 200):

    # Calculate end-effector position
        X = a[0] * np.cos(theta1) + a[1] * np.cos(theta1 + theta2)
        Y = a[0] * np.sin(theta1) + a[1] * np.sin(theta1 + theta2)
    # Store the end-effector position
        endEffectorPositions.append([X, Y])
 
 # Convert the list to a NumPy array
endEffectorPositions = np.array(endEffectorPositions)
 
# Plot the workspace
plt.plot(endEffectorPositions[:, 0], endEffectorPositions[:, 1])
plt.title('Workspace of 2R Manipulator')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.axis('equal')
plt.show()