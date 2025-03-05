import socket


def main():
    # Укажите IP-адрес и порт Raspberry Pi, где запущен эхо-сервер
    local_ip = "0.0.0.0"  # Привязываем к всем интерфейсам
    local_port = 5005          # Локальный порт для приёма сообщений
    target_ip = "192.168.1.47"  # IP-адрес получателя (например, Raspberry Pi)
    target_port = 5005  # Порт получателя (должен совпадать с портом, на котором работает сервер на Raspberry Pi)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Привязываем сокет к локальному адресу и порту для приёма сообщений
    sock.bind((local_ip, local_port))
    print(f"UDP socket bound on {local_ip}:{local_port}")
    print(f"Target: {target_ip}:{target_port}")

    try:
        while True:
            # Получаем сообщение для отправки с клавиатуры
            message = input("Enter message to send (or 'q' to quit): ")
            if message.lower() == 'q':
                break

            # Отправляем сообщение
            sock.sendto(message.encode('utf-8'), (target_ip, target_port))
            print(f"Message sent to {target_ip}:{target_port}")

            # Ожидаем ответа от получателя (например, эхо-сообщения)
            sock.settimeout(5.0)  # задаем таймаут ожидания в 5 секунд
            try:
                data, addr = sock.recvfrom(1024)
                print(f"Received from {addr}: {data.decode('utf-8')}")
            except socket.timeout:
                print("No response received within timeout.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sock.close()


if __name__ == "__main__":
    main()