import os
import numpy as np
import pandas as pd

# Simulate sensor data collection for grip classification
# Each sample contains a 3-second time-series window at 50Hz = 150 timesteps

SAMPLE_COUNT = 100  # Number of samples for both successful and failed grips
TIMESTEPS = 150
# SENSORS = ["fsr", "imu_pitch", "imu_roll", "spring_deflection", "drill_current"]
SENSORS = ["fsr", "imu_pitch", "imu_roll", "drill_current"]

# Create random data with some patterns to simulate two classes
def generate_sensor_data(success=True):
    base = np.random.normal(loc=0.5 if success else 0.2, scale=0.05 if success else 0.1, size=(TIMESTEPS, len(SENSORS)))
    noise = np.random.normal(0, 0.01, size=base.shape)
    print(base.shape)
    return base + noise

# Generate dataset
data = []
labels = []

for _ in range(SAMPLE_COUNT):
    data.append(generate_sensor_data(success=True))
    labels.append(1)
    print("data len:", len(data))
    data.append(generate_sensor_data(success=False))
    labels.append(0)

# Convert to 3D array (samples, timesteps, features)
data = np.array(data)
labels = np.array(labels)
print("data shape:", data.shape)
print("labels shape:", labels.shape)

# Save as .npz file
output_dir = "/home/sml/Documents/artemisDrill/script/horizontal/drm_rt"
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(os.path.join(output_dir, "grip_data1.npz"), data=data, labels=labels)
