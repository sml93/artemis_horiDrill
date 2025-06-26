# main.py
from getSensor import SensorReader

class sensingReader(SensorReader):
    def handle_line(self, line):
        # Custom processing of each line from the IMU
        # print(f"[Parsed IMU]: {line}")
        # Split line by commas
        parts = line.split(',')
        if len(parts) == 4:
            try:
                x, y, z, w = map(float, parts)
                print(f"X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}, Force(g): {w:.3f}")
            except ValueError:
                print(f"[Invalid float values] {line}")
        else:
            print(f"[Error] {line}")
        return x, y, z, w

if __name__ == "__main__":
    reader = sensingReader()
    reader.start_reading()
