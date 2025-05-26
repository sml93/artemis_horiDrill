# main.py
from getIMU import BNO055Reader

class MyIMUReader(BNO055Reader):
    def handle_line(self, line):
        # Custom processing of each line from the IMU
        print(f"[Parsed IMU]: {line}")

if __name__ == "__main__":
    reader = MyIMUReader()
    reader.start_reading()
