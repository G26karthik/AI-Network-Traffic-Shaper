# traffic_simulation/traffic_generator.py

import socket
import threading
import time
import random

def simulate_voip_traffic(host='127.0.0.1', port=5555):
    """Simulate low-latency UDP VoIP traffic"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(100):
        message = f"VoIP Packet {i}".encode()
        sock.sendto(message, (host, port))
        time.sleep(0.05)  # simulate 20 packets/sec
    sock.close()

def simulate_ftp_traffic(host='127.0.0.1', port=6666):
    """Simulate bulk file transfer (FTP-like)"""
    sock = socket.socket()
    sock.connect((host, port))
    data = b'x' * 1024 * 50  # 50 KB per chunk
    for _ in range(100):
        sock.send(data)
        time.sleep(0.1)
    sock.close()

def simulate_http_traffic(host='127.0.0.1', port=7777):
    """Simulate short TCP requests (HTTP-like)"""
    for _ in range(50):
        sock = socket.socket()
        sock.connect((host, port))
        request = f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n"
        sock.send(request.encode())
        sock.close()
        time.sleep(0.2)

# Create dummy servers
def start_udp_server(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"[UDP] {addr} → {data.decode()}")

def start_tcp_server(port):
    sock = socket.socket()
    sock.bind(('0.0.0.0', port))
    sock.listen(5)
    while True:
        conn, addr = sock.accept()
        total_bytes = 0
        while True:
            data = conn.recv(4096)
            if not data:
                break
            total_bytes += len(data)
        print(f"[TCP] {addr} → {total_bytes} bytes")
        conn.close()

if __name__ == "__main__":
    # Start servers in background
    threading.Thread(target=start_udp_server, args=(5555,), daemon=True).start()
    threading.Thread(target=start_tcp_server, args=(6666,), daemon=True).start()
    threading.Thread(target=start_tcp_server, args=(7777,), daemon=True).start()

    time.sleep(2)  # Give servers time to start

    # Simulate each type of traffic
    simulate_voip_traffic()
    simulate_ftp_traffic()
    simulate_http_traffic()
