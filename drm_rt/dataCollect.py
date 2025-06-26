import time
import numpy as np

from sensorRead import sensingReader

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
            fsr,                # Normalized FSR        to detect grip strength
            imu_pitch,                # degrees or radians    to read imu pitch
            imu_roll,                 # degrees or radians    to read imu roll
            # get_spring_deflection(),  # mm or normalized
            get_drill_motor_revolutions() # RPM                  to determine how much the 
        ]
        sensor_data.append(sample)
        time.sleep(1 / freq_hz)
    return np.array(sensor_data)