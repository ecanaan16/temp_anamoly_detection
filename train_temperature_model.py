import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate simulated room temperature data
np.random.seed(42)

n_points = 1200
time = np.arange(n_points)

temps = 24 + 0.8 * np.sin(2 * np.pi * time / 200) + np.random.normal(0, 0.15, n_points)

temps_with_anomalies = temps.copy()
temps_with_anomalies[300] += 4.0
temps_with_anomalies[700] -= 5.0
temps_with_anomalies[950] += 3.5

# 2. Create sliding windows
WINDOW_SIZE = 10

def create_windows(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y = create_windows(temps, WINDOW_SIZE)

# 3. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 4. Normalize
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))

# 5. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Train
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=50,
    batch_size=16,
    verbose=1
)

# 7. Validation error
val_pred_scaled = model.predict(X_val_scaled, verbose=0)
val_pred = y_scaler.inverse_transform(val_pred_scaled).flatten()

val_error = np.abs(val_pred - y_val)
threshold = np.mean(val_error) + 3 * np.std(val_error)

print("Threshold:", threshold)

# 8. Test on anomalous data
X_test, y_test = create_windows(temps_with_anomalies, WINDOW_SIZE)
X_test_scaled = x_scaler.transform(X_test)

test_pred_scaled = model.predict(X_test_scaled, verbose=0)
test_pred = y_scaler.inverse_transform(test_pred_scaled).flatten()

test_error = np.abs(test_pred - y_test)
anomalies = test_error > threshold

anomaly_indices = np.where(anomalies)[0] + WINDOW_SIZE
print("Detected anomaly indices:", anomaly_indices)

# 9. Plot temperature and detected anomalies
plt.figure(figsize=(12, 5))
plt.plot(temps_with_anomalies, label="Temperature")
plt.scatter(anomaly_indices, temps_with_anomalies[anomaly_indices], label="Detected anomalies")
plt.legend()
plt.title("Temperature Anomaly Detection")
plt.xlabel("Time step")
plt.ylabel("Temperature (C)")
plt.show()

# 10. Plot error
plt.figure(figsize=(12, 4))
plt.plot(test_error, label="Prediction Error")
plt.axhline(threshold, linestyle='--', label="Threshold")
plt.legend()
plt.title("Prediction Error")
plt.xlabel("Window index")
plt.ylabel("Absolute Error")
plt.show()

# 11. Save model
model.save("temperature_model.h5")