from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a sample dataset
data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the data and transform the data
scaled_data = scaler.fit_transform(data)

print("Original Data:")
print(data)

print("\nScaled Data:")
print(scaled_data)