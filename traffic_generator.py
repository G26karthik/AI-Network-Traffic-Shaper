"""
traffic_generator.py

Generate synthetic network traffic for different application types using scapy:
- VoIP-like (UDP bursts) on port 5555
- FTP-like (TCP control/data-ish) on port 6666
- HTTP-like (TCP request-ish) on port 7777

Notes (Windows):
- Run PowerShell as Administrator for raw packet sending.
- Having Wireshark/Npcap installed helps with sending and capturing.

Usage examples (PowerShell):
  # Generate VoIP-like traffic at 50 packets/sec for 10s to localhost
  .\traffic_generator.py --type voip --duration 10 --pps 50

  # Generate all three types for 15s to a target IP
  .\traffic_generator.py --type all --duration 15 --dst 127.0.0.1 --pps 30

"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional
import socket

try:
    # Import scapy lazily to let --help work even if scapy isn't installed
    from scapy.all import IP, UDP, TCP, Raw, RandShort, send
except Exception as e:  # pragma: no cover - only hit when scapy missing
    IP = UDP = TCP = Raw = RandShort = send = None  # type: ignore
    _import_error: Optional[Exception] = e
else:
    _import_error = None


VOIP_PORT = 5555
FTP_PORT = 6666
HTTP_PORT = 7777

# --- Socket-based senders (defined early to be available in main) ---
def send_voip_socket(dst: str, pps: int, duration: float, payload_size: int = 160) -> None:
    """Send UDP datagrams using sockets (more reliable visibility on Windows)."""
    interval = 1.0 / max(1, pps)
    end_t = time.time() + duration
    sent = 0
    print(f"[+] VoIP(socket): UDP -> {dst}:{VOIP_PORT} @ {pps} pps for {duration}s (payload={payload_size} bytes)")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        payload = os.urandom(payload_size)
        while time.time() < end_t:
            try:
                s.sendto(payload, (dst, VOIP_PORT))
                sent += 1
            except Exception:
                pass
            time.sleep(interval)
    print(f"[+] VoIP(socket) done. Sent ~{sent} datagrams")


def send_tcp_like_socket(dst: str, pps: int, duration: float, dport: int) -> None:
    """Attempt TCP connects repeatedly to generate SYN traffic (and payload if connected)."""
    interval = 1.0 / max(1, pps)
    end_t = time.time() + duration
    attempts = 0
    print(f"[+] TCP-like(socket): -> {dst}:{dport} @ {pps} conn-attempts/sec for {duration}s")
    while time.time() < end_t:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                s.connect_ex((dst, dport))
                try:
                    s.send(b"HELLO")
                except Exception:
                    pass
        except Exception:
            pass
        attempts += 1
        time.sleep(interval)
    print(f"[+] TCP-like(socket) done. Attempts ~{attempts}")


def is_admin() -> bool:
    if os.name != "nt":
        return os.geteuid() == 0  # type: ignore[attr-defined]
    try:
        import ctypes  # windows-only check

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _ensure_scapy():
    if _import_error is not None:
        raise RuntimeError(
            "Scapy import failed. Ensure scapy is installed in your venv (pip install scapy)."
        ) from _import_error


def send_voip(dst: str, pps: int, duration: float, payload_size: int = 160) -> None:
    """Send UDP packets to emulate VoIP RTP-like stream.

    - Small fixed-size payloads, steady rate.
    """
    _ensure_scapy()
    interval = 1.0 / max(1, pps)
    end_t = time.time() + duration
    sent = 0
    print(f"[+] VoIP: UDP -> {dst}:{VOIP_PORT} @ {pps} pps for {duration}s (payload={payload_size} bytes)")
    while time.time() < end_t:
        pkt = IP(dst=dst) / UDP(sport=RandShort(), dport=VOIP_PORT) / Raw(os.urandom(payload_size))
        send(pkt, verbose=0)
        sent += 1
        time.sleep(interval)
    print(f"[+] VoIP done. Sent ~{sent} packets")


def send_ftp_like(dst: str, pps: int, duration: float) -> None:
    """Send TCP packets to emulate FTP control/data activity.

    We craft a small sequence of TCP flags with tiny payloads. No real 3-way handshake is required.
    """
    _ensure_scapy()
    interval = 1.0 / max(1, pps)
    end_t = time.time() + duration
    sent = 0
    print(f"[+] FTP-like: TCP -> {dst}:{FTP_PORT} @ {pps} pps for {duration}s")
    while time.time() < end_t:
        sport = int(RandShort())
        seq = int(time.time() * 1000) & 0xFFFFFFFF
        # SYN
        send(IP(dst=dst) / TCP(sport=sport, dport=FTP_PORT, flags="S", seq=seq), verbose=0)
        time.sleep(interval / 3)
        # PUSH/ACK with small payload
        send(
            IP(dst=dst)
            / TCP(sport=sport, dport=FTP_PORT, flags="PA", seq=seq + 1)
            / Raw(b"USER test\r\n"),
            verbose=0,
        )
        time.sleep(interval / 3)
        # FIN
        send(IP(dst=dst) / TCP(sport=sport, dport=FTP_PORT, flags="F", seq=seq + 2), verbose=0)
        sent += 3
        time.sleep(interval / 3)
    print(f"[+] FTP-like done. Sent ~{sent} packets")


def send_http_like(dst: str, pps: int, duration: float) -> None:
    """Send TCP packets to emulate a simple HTTP request/response flow.

    Crafted packets: SYN, PSH/ACK with GET line, FIN. Repeats at given rate.
    """
    _ensure_scapy()
    interval = 1.0 / max(1, pps)
    end_t = time.time() + duration
    sent = 0
    print(f"[+] HTTP-like: TCP -> {dst}:{HTTP_PORT} @ {pps} pps for {duration}s")
    while time.time() < end_t:
        sport = int(RandShort())
        seq = int(time.time() * 1000) & 0xFFFFFFFF
        # SYN
        send(IP(dst=dst) / TCP(sport=sport, dport=HTTP_PORT, flags="S", seq=seq), verbose=0)
        time.sleep(interval / 3)
        # GET request
        payload = b"GET / HTTP/1.1\r\nHost: example\r\n\r\n"
        send(
            IP(dst=dst)
            / TCP(sport=sport, dport=HTTP_PORT, flags="PA", seq=seq + 1)
            / Raw(payload),
            verbose=0,
        )
        time.sleep(interval / 3)
        # FIN
        send(IP(dst=dst) / TCP(sport=sport, dport=HTTP_PORT, flags="F", seq=seq + 2), verbose=0)
        sent += 3
        time.sleep(interval / 3)
    print(f"[+] HTTP-like done. Sent ~{sent} packets")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic traffic (VoIP/FTP/HTTP) using scapy")
    p.add_argument("--type", choices=["voip", "ftp", "http", "all"], default="all", help="Traffic type")
    p.add_argument("--dst", default="127.0.0.1", help="Destination IP address (default 127.0.0.1)")
    p.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (default 10)")
    p.add_argument("--pps", type=int, default=30, help="Packets per second target rate (default 30)")
    p.add_argument("--voip-port", type=int, default=VOIP_PORT, help="VoIP UDP port (default 5555)")
    p.add_argument("--ftp-port", type=int, default=FTP_PORT, help="FTP TCP port (default 6666)")
    p.add_argument("--http-port", type=int, default=HTTP_PORT, help="HTTP TCP port (default 7777)")
    p.add_argument("--method", choices=["scapy", "socket"], default=("socket" if os.name == "nt" else "scapy"), help="Packet send method")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Allow overriding ports via args
    global VOIP_PORT, FTP_PORT, HTTP_PORT
    VOIP_PORT = int(args.voip_port)
    FTP_PORT = int(args.ftp_port)
    HTTP_PORT = int(args.http_port)

    if os.name == "nt" and not is_admin() and args.method == "scapy":
        print("[!] Warning: Not running as Administrator. Scapy raw packet sending may fail on Windows. Consider --method socket.")

    if args.method == "scapy" and _import_error is not None:
        print("[x] Scapy not available. Install it or use --method socket")
        return 2

    try:
        if args.method == "scapy":
            if args.type in ("voip", "all"):
                send_voip(args.dst, args.pps, args.duration)
            if args.type in ("ftp", "all"):
                send_ftp_like(args.dst, args.pps, args.duration)
            if args.type in ("http", "all"):
                send_http_like(args.dst, args.pps, args.duration)
        else:
            # Socket-based fallback for Windows reliability
            if args.type in ("voip", "all"):
                send_voip_socket(args.dst, args.pps, args.duration)
            if args.type in ("ftp", "all"):
                send_tcp_like_socket(args.dst, args.pps, args.duration, FTP_PORT)
            if args.type in ("http", "all"):
                send_tcp_like_socket(args.dst, args.pps, args.duration, HTTP_PORT)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        return 130
    except Exception as e:
        print(f"[x] Error during generation: {e}")
        return 1

    print("[✓] Traffic generation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
