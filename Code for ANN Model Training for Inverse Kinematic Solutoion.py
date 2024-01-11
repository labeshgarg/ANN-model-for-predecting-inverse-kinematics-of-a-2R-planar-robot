 
'''ANN Model Training on given data set for predicting inverse kinematic solution'''
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

#Load the dataset
dataset_path = '/mydataset.csv'
data = pd.read_csv(dataset_path)

# Extract input features (X, Y)
X = data[['X', 'Y']].values

# Extract target values (Theta1_deg, Theta2_deg)
y = data[['Theta1_deg', 'Theta2_deg']].values

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features (X) for better model training
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
 
# Build the ANN model with increased complexity and dropout
model = Sequential()
model.add(Dense(128, input_dim=2, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='linear')) 

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=40, batch_size=20, verbose=1, validation_data=(X_test_scaled, y_test))

# Evaluate the model on the testing data
loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Mean Squared Error on Testing Data: {loss}')

# Make predictions
predictions = model.predict(X_test_scaled)

# Display some predictions and actual values
for i in range(5):  
    print(f"Prediction {i + 1}: X={X_test[i][0]:.2f}, Y={X_test[i][1]:.2f} -> Predicted Theta1={predictions[i][0]:.2f}, Predicted Theta2={predictions[i][1]:.2f} (Actual Theta1={y_test[i][0]:.2f}, Actual Theta2={y_test[i][1]:.2f})")

# Calculate the error (difference) between actual theta values and predicted theta values
theta_error = y_test - predictions

# Calculate the mean and standard deviation of the error for Theta1 and Theta2
mean_theta_error = np.mean(theta_error, axis=0)
std_deviation_theta_error = np.std(theta_error, axis=0)

print(f'Mean Theta Error (Theta1, Theta2): {mean_theta_error}')
print(f'Standard Deviation of Theta Error (Theta1, Theta2): {std_deviation_theta_error}')

# Plot the training and validation loss over epochs
plt.figure(figsize=(12, 6))

# Plot the loss history
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Loss Over Time')
plt.legend()

# Plot the Theta error and standard deviation
plt.subplot(1, 2, 2)
plt.bar(['Theta1', 'Theta2'], mean_theta_error, yerr=std_deviation_theta_error, capsize=10, align='center')
plt.ylabel('Mean Error')
plt.title('Mean Error and Standard Deviation for Theta1 and Theta2')

plt.tight_layout()
