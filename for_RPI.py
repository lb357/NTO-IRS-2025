from logging import exception

from dotenv import load_dotenv
from robot import Robot
import socket


load_dotenv('.env')


def solve():
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    robot = Robot()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"UDP сервер запущен и слушает на порту {UDP_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')
        try:
            left_voltage, right_voltage = map(int, message.split())
            robot.send_voltage(left_voltage, right_voltage)
            print(f"Установлена мощность моторов: {message}")
        except Exception as error:
            print(error)
        sock.sendto(data, addr)


if __name__ == '__main__':
    solve()
