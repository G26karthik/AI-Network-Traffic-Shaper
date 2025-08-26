"""
capture_features.py

Live-capture network traffic using PyShark/TShark and extract features for ML.
Labels are assigned based on destination port:
  5555 -> VoIP (UDP)
  6666 -> FTP (TCP)
  7777 -> HTTP (TCP)

Output: CSV with columns: timestamp, protocol, length, src_ip, dst_ip, src_port, dst_port, label

Windows notes:
- Run PowerShell as Administrator to allow capture on most interfaces.
- Use --list to discover interface names (wraps `tshark -D`).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List

import pyshark


PORT_LABELS: Dict[int, str] = {5555: "VoIP", 6666: "FTP", 7777: "HTTP"}


def list_interfaces() -> List[str]:
    try:
        out = subprocess.check_output(["tshark", "-D"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print(f"[x] Failed to run 'tshark -D': {e}")
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # Example: '1. Ethernet' or '2. \Device\NPF_{GUID} (Npcap Loopback Adapter)'
    return lines


def get_label_from_port(dst_port: str) -> str:
    try:
        p = int(dst_port)
    except Exception:
        return "Unknown"
    return PORT_LABELS.get(p, "Other")


def get_ips(pkt) -> tuple[str, str]:
    src_ip = dst_ip = "N/A"
    try:
        if hasattr(pkt, "ip") and hasattr(pkt.ip, "src"):
            src_ip = pkt.ip.src
            dst_ip = pkt.ip.dst
        elif hasattr(pkt, "ipv6") and hasattr(pkt.ipv6, "src"):
            src_ip = pkt.ipv6.src
            dst_ip = pkt.ipv6.dst
    except Exception:
        pass
    return src_ip, dst_ip


def capture_and_extract(
    interface: str,
    duration: int,
    output_csv: str,
    display_filter: str | None = None,
    bpf_filter: str | None = None,
) -> int:
    # Preflight: verify tshark availability
    try:
        v = subprocess.check_output(["tshark", "-v"], stderr=subprocess.STDOUT, text=True)
        if v:
            first = v.splitlines()[0]
            print(f"[i] Using {first}")
    except Exception as e:
        raise RuntimeError("tshark not available. Ensure Wireshark/TShark is installed and in PATH.") from e
    print(f"[+] Starting capture on '{interface}' for {duration}s...")
    if display_filter is None:
        display_filter = "tcp.port==6666 or tcp.port==7777 or udp.port==5555"
    if bpf_filter is None:
        bpf_filter = "tcp port 6666 or tcp port 7777 or udp port 5555"

    # Use tshark auto-stop to guarantee termination; -l for line-buffered output
    cap = pyshark.LiveCapture(
        interface=interface,
        display_filter=display_filter,
        bpf_filter=bpf_filter,
        custom_parameters=["-a", f"duration:{duration}", "-l"],
    )

    rows: List[dict] = []
    try:
        # Add a small grace to allow pyshark to flush remaining packets after tshark exits
        cap.sniff(timeout=duration + 3)
        for pkt in cap:
            try:
                timestamp = float(pkt.sniff_timestamp)
                protocol = pkt.highest_layer
                length = int(pkt.length)

                src_ip, dst_ip = get_ips(pkt)
                t_layer = getattr(pkt, "transport_layer", None)
                if not t_layer:
                    continue
                src_port = pkt[t_layer].srcport if hasattr(pkt[t_layer], "srcport") else "N/A"
                dst_port = pkt[t_layer].dstport if hasattr(pkt[t_layer], "dstport") else "N/A"

                label = get_label_from_port(dst_port)

                rows.append(
                    {
                        "timestamp": timestamp,
                        "protocol": protocol,
                        "length": length,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "label": label,
                    }
                )
            except Exception:
                # Skip malformed/incomplete packets
                continue
    except Exception as e:
        # Make sure we don't hang on unexpected sniff issues
        print(f"[x] Capture loop error: {e}")
    finally:
        cap.close()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Write CSV
    new_file = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "protocol",
                "length",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "label",
            ],
        )
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[+] Captured {len(rows)} packets. Appended to {output_csv}")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Capture traffic and extract features into dataset.csv")
    p.add_argument("--interface", "-i", default=None, help="Capture interface name (use --list to show)")
    p.add_argument("--duration", "-d", type=int, default=15, help="Capture duration in seconds (default 15)")
    p.add_argument("--output", "-o", default="dataset.csv", help="Output CSV path (default dataset.csv)")
    p.add_argument("--list", action="store_true", help="List available interfaces and exit")
    p.add_argument("--filter", default=None, help="Optional display filter override (Wireshark syntax)")
    p.add_argument("--bpf", default=None, help="Optional capture (BPF) filter, e.g., 'tcp port 80' ")
    p.add_argument("--no-filter", action="store_true", help="Disable both display and BPF filters")
    args = p.parse_args(argv)

    if args.list:
        print("Available interfaces (from 'tshark -D'):")
        for ln in list_interfaces():
            print("  ", ln)
        return 0

    interface = args.interface
    if not interface:
        # Try to pick a reasonable default on Windows
        all_if = list_interfaces()
        preferred = None
        for ln in all_if:
            if "Loopback" in ln or "Ethernet" in ln or "Wi-Fi" in ln or "WiFi" in ln:
                preferred = ln
                break
        if preferred:
            # Extract interface name/number; tshark accepts the number (prefix before dot)
            # e.g., '1. Ethernet' -> '1'
            num = preferred.split(".", 1)[0].strip()
            interface = num if num.isdigit() else preferred
            print(f"[i] No interface provided. Using: {preferred}")
        else:
            print("[x] Could not auto-select an interface. Use --list then provide --interface <name|index>.")
            return 2

    try:
        capture_and_extract(
            interface=interface,
            duration=args.duration,
            output_csv=args.output,
            display_filter=(None if args.no_filter else args.filter),
            bpf_filter=(None if args.no_filter else args.bpf),
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        return 130
    except Exception as e:
        print(f"[x] Capture failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
