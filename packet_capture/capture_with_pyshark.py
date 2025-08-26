# packet_capture/capture_with_pyshark.py

import pyshark

def capture_packets(interface='Ethernet', output_file='captured_traffic.pcapng', duration=30):
    print(f"[+] Starting capture on {interface} for {duration} seconds...")
    capture = pyshark.LiveCapture(interface=interface, output_file=output_file)
    capture.sniff(timeout=duration)
    print(f"[+] Capture complete. Packets saved to {output_file}")

if __name__ == "__main__":
    capture_packets()
