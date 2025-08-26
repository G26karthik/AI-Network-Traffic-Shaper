"""
predict_and_shape.py

Live-predict traffic type on captured packets and optionally shape (block) by label.

Captures with PyShark, extracts minimal features, runs the saved pipeline from train_model.py,
and (optionally) applies simple Windows Firewall rules to block ports associated with predictions.

Windows admin rights required to modify firewall rules. Shaping is optional and opt-in via CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List, Tuple

import joblib
import pyshark
import pandas as pd


PORT_LABELS: Dict[int, str] = {5555: "VoIP", 6666: "FTP", 7777: "HTTP"}


def get_features_from_pkt(pkt) -> dict | None:
    try:
        protocol = pkt.highest_layer
        length = int(pkt.length)
        t_layer = getattr(pkt, "transport_layer", None)
        if not t_layer:
            return None
        src_port = int(pkt[t_layer].srcport) if hasattr(pkt[t_layer], "srcport") else None
        dst_port = int(pkt[t_layer].dstport) if hasattr(pkt[t_layer], "dstport") else None
        if src_port is None or dst_port is None:
            return None
        return {
            "protocol": protocol,
            "length": length,
            "src_port": src_port,
            "dst_port": dst_port,
        }
    except Exception:
        return None


def block_port_windows(port: int, name: str, protocol: str) -> None:
    # Create an outbound and inbound block rule for the target port and protocol (TCP/UDP)
    for direction in ("in", "out"):
        rule_name = f"AI-Traffic-Shaper {name} {protocol} {direction} port {port}"
        try:
            subprocess.run(
                [
                    "netsh",
                    "advfirewall",
                    "firewall",
                    "add",
                    "rule",
                    f"name={rule_name}",
                    "dir=" + direction,
                    "action=block",
                    f"protocol={protocol}",
                    f"localport={port}",
                    "enable=yes",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"[x] Failed to add firewall rule: {e}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live prediction and optional shaping of traffic")
    p.add_argument("--model", default="traffic_model.pkl", help="Path to saved sklearn Pipeline")
    p.add_argument("--interface", "-i", required=True, help="Capture interface (index or name)")
    p.add_argument("--duration", "-d", type=int, default=15, help="Duration to run (seconds)")
    p.add_argument("--shape", action="store_true", help="Enable shaping (block predicted ports via Windows Firewall)")
    p.add_argument(
        "--filter",
        default="tcp.port==6666 or tcp.port==7777 or udp.port==5555",
        help="Display filter",
    )
    args = p.parse_args(argv)

    try:
        pipe = joblib.load(args.model)
    except Exception as e:
        print(f"[x] Failed to load model: {e}")
        return 2

    cap = pyshark.LiveCapture(interface=args.interface, display_filter=args.filter)
    blocked: set[Tuple[str, int]] = set()
    print("[+] Starting live prediction...")
    try:
        cap.sniff(timeout=args.duration)
        for pkt in cap:
            feats = get_features_from_pkt(pkt)
            if not feats:
                continue
            # Prepare a single-row prediction input
            X = pd.DataFrame([feats])
            pred = pipe.predict(X)[0]
            print(f"Predicted: {pred} for dst_port={feats['dst_port']}, protocol={feats['protocol']}")
            if args.shape:
                # Map predicted label to port (simple mapping) and associated protocol
                label_to_port = {v: k for k, v in PORT_LABELS.items()}
                port = label_to_port.get(str(pred))
                proto = "UDP" if str(pred) == "VoIP" else "TCP"
                key = (proto, port) if port else None
                if port and key not in blocked:
                    print(f"[i] Applying firewall block for {pred} on {proto} port {port} (once per run)")
                    block_port_windows(port, str(pred), proto)
                    blocked.add(key)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        return 130
    finally:
        cap.close()

    print("[✓] Live prediction finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
