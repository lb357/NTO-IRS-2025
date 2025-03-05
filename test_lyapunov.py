from logging import exception

from dotenv import load_dotenv
from robot import Robot
import socket
import math
speed = 0.5
turn = 1

load_dotenv('.env')

def lyapunov_calcvelocity(target_x, target_y, current_x, current_y, current_theta, max_speed, k2):
    normalized_x = target_x-current_x
    normalized_y = target_y-current_y
    angle_error = math.atan2(normalized_y, normalized_x)
    dist = (normalized_x**2+normalized_y**2)**0.5
    linear_speed = max_speed*math.tanh(dist)*math.cos(angle_error)
    angle_speed = (max_speed*math.tanh(dist)*math.cos(angle_error)
                   * math.sin(angle_error))/dist+k2*angle_error
    return linear_speed, angle_speed


def velocity_to_voltage(linear_s, angle_s):
    v = linear_s * speed
    w = angle_s * turn
    wl = 0.5 * (v - w)
    wr = 0.5 * (v + w)
    left_voltage = int(wl * 255)
    right_voltage = int(wr * 255)
    return left_voltage, right_voltage

def solve():
    UDP_IP = "192.168.1.47"
    UDP_PORT = 5005
    robot = Robot()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"UDP сервер запущен и слушает на порту {UDP_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')
        try:
            left_voltage,right_voltage=velocity_to_voltage(lyapunov_calcvelocity(*list(map(int, message.split()))))
            # left_voltage, right_voltage = map(int, message.split())
            robot.send_voltage(left_voltage, right_voltage)
            print(f"Установлена мощность моторов: {left_voltage},{right_voltage}")
        except Exception as error:
            print(error)
        sock.sendto(data, addr)

if __name__ == '__main__':
    solve()