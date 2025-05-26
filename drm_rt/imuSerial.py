import serial
import time
import threading
import sys

arduino_port = '/dev/ttyACM0'
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # wait for Arduino reset

running = True

def key_listener():
    global running
    print("Press 'r' to recalibrate. Press Ctrl+C to exit.")
    while running:
        key = sys.stdin.read(1)
        if key == 'r':
            print("[Python] Sending 'r' to trigger recalibration...")
            ser.write(b'r')  # Send byte to Arduino

# Run key listener in a separate thread
threading.Thread(target=key_listener, daemon=True).start()

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print(line)
except KeyboardInterrupt:
    print("\nExiting...")
    running = False
finally:
    ser.close()

    

# import serial
# import time
# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # Serial config
# arduino_port = '/dev/ttyACM0'  # Adjust as needed
# baud_rate = 115200

# # ser = serial.Serial(arduino_port, baud_rate, timeout=1)
# ser = serial.Serial(arduino_port, 115200)
# time.sleep(2)  # Let the Arduino reset

# print("Reading quaternion data and converting to Euler angles...")

# try:
#     while True:
#         line = ser.readline().decode('utf-8').strip()
#         if line:
#             try:
#                 # Parse quaternion
#                 w, x, y, z = map(float, line.split(','))
                
#                 # Convert to Euler (degrees)
#                 r = R.from_quat([x, y, z, w])  # Note: scipy uses [x, y, z, w]
#                 roll, pitch, yaw = r.as_euler('xyz', degrees=True)
                
#                 print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
#             except ValueError:
#                 print(f"Invalid data: {line}")
# except KeyboardInterrupt:
#     print("Stopped.")
# finally:
#     ser.close()
