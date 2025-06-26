import torch
import numpy as np
from model import GripLSTM  # assumes model class is saved in model.py
from sensorRead import sensingReader
import time

# Load trained model
input_size = 5       # fsr, imu_pitch, imu_roll, spring_deflection, drill_current
hidden_size = 64
num_layers = 1
num_classes = 2

model = GripLSTM(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load("grip_lstm_model.pth"))  # load trained weights
model.eval()

# Function to collect real sensor data
def read_sensor_window(duration_sec=3, freq_hz=50):
    steps = int(duration_sec * freq_hz)
    sensor_data = []
    reader = sensingReader()
    imu_roll, imu_pitch, z, fsr = reader.start_reading()
    reader.start_reading()
    for _ in range(steps):
        # Replace this with actual sensor reading code
        sample = [
            fsr,                        # Normalized FSR        to detect grip strength
            # get_imu_pitch(),          # degrees or radians    to read imu pitch
            # get_imu_roll(),           # degrees or radians    to read imu roll
            imu_pitch,                  # degrees or radians    to read imu pitch
            imu_roll,                   # degrees or radians    to read imu roll
            # get_spring_deflection(),  # mm or normalized
            get_drill_motor_current()   # Amps                  to a
        ]
        sensor_data.append(sample)
        time.sleep(1 / freq_hz)
    return np.array(sensor_data)

# Preprocess and run inference
def classify_grip(sensor_window):
    tensor_input = torch.tensor(sensor_window, dtype=torch.float32).unsqueeze(0)  # shape (1, T, F)
    with torch.no_grad():
        output = model(tensor_input)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()  # 0 = unstable, 1 = stable

# Example usage
if __name__ == "__main__":
    print("Waiting for perch contact...")
    while not contact_detected():   # use IMU or pressure spike to detect contact
        time.sleep(0.1)

    print("Perch detected. Collecting sensor data...")
    window = read_sensor_window()
    result = classify_grip(window)

    if result == 1:
        print("Grip is stable. Proceed with drilling.")
        start_drilling_sequence()   # trigger drilling command
    else:
        print("Grip is unstable. Aborting drilling.")
        retry_perching_or_adjust()  # readjust pose or reattempt to grip
