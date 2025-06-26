# getSensor.py
import serial
import threading
import sys
import time

class SensorReader:
    def __init__(self, port='/dev/ttyACM0', baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.running = False
        time.sleep(2)  # Allow Arduino to reset

    def _key_listener(self):
        print("Press 'r' to recalibrate. Press Ctrl+C to stop.")
        while self.running:
            key = sys.stdin.read(1)
            if key == 'r':
                print("[Python] Sending 'r' to trigger recalibration...")
                self.ser.write(b'r')

    def start_reading(self):
        self.running = True
        # Start key listener thread
        threading.Thread(target=self._key_listener, daemon=True).start()

        try:
            while self.running:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    self.handle_line(line)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Stopping...")
            self.stop()
        finally:
            self.ser.close()

    def handle_line(self, line):
        """Override this method in subclasses or external code."""
        print(line)
        pass

    def stop(self):
        self.running = False
