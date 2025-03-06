from robot import Robot
import socket
from math import atan2, pi, sqrt


def get_pos_delta(xc, yc, xt, yt, th):
    p = sqrt((xt-xc)**2 + (yt-yc)**2)
    a = atan2(yt-yc, xt-xc) - th
    if a > pi:
        a = a - 2*pi
    if a < -pi:
        a = a + 2*pi
    return p, a


def get_wheel_const(p, a, k1, k2, k3):
    if abs(a) >= k1:
        return -a * k2, a * k2
    else:
        return p * k3, p * k3


def clamp_v(v, min_v, max_v, kc=2):
    if v < -min_v / kc:
        return max(-max_v, min(v, -min_v))
    elif v > min_v / kc:
        return max(min_v, min(v, max_v))
    else:
        return 0


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
            xr, yr, xt, yt, ang = map(float, message.split())
            a, p = get_pos_delta(xr, yr, xt, yt, ang)
            print("Получены данные")
            if p <= 0.1:
                robot.send_voltage(0, 0)
                continue
            kl, kr = 0.0003, 0.00044
            ka, kd = 1, 1
            const = 15 * pi / 180

            if abs(a) >= const:
                vr = ka * a
                vl = -ka * a
            else:
                vr = kd * p
                vl = kd * p

            sl = int(vl / kl)
            sr = int(vr / kr)
            robot.send_voltage(sl, sr)

        except Exception as error:
            print(error)




if __name__ == '__main__':
    solve()
