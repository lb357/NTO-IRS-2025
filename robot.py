import struct
import serial


class Robot:
    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=9600):
        if serial_port is None:
            raise ValueError("Для serial-режима необходимо указать serial_port")
        self.ser = serial.Serial(serial_port, baudrate, timeout=1)

    @staticmethod
    def voltage_to_bytes_msg(left_voltage: int, right_voltage: int) -> bytes:
        orientation = 0

        if left_voltage < 0 and right_voltage < 0:
            orientation = 1

        elif left_voltage >= 0 > right_voltage:
            orientation = 2

        elif left_voltage < 0 <= right_voltage:
            orientation = 3

        left_motor = abs(left_voltage) if abs(left_voltage) <= 255 else 255
        right_motor = abs(right_voltage) if abs(right_voltage) <= 255 else 255
        message = struct.pack("BBB", left_motor, right_motor, orientation)

        return message

    def send_voltage(self, left_voltage: int, right_voltage: int):
        message = self.voltage_to_bytes_msg(left_voltage, right_voltage)
        self.ser.write(message)
