"""
predict_and_shape.py

Live-predict traffic type on captured packets and optionally shape (block) by label.

Captures with PyShark, extracts minimal features, runs the saved pipeline from train_model.py,
and (optionally) applies simple Windows Firewall rules to block ports associated with predictions.

⚠️  WARNING: Traffic shaping modifies Windows Firewall rules and can block legitimate traffic!
    - Only use on isolated lab networks
    - Always test with --dry-run first
    - Use scripts/cleanup_firewall_rules.ps1 to remove rules if needed

Windows admin rights required to modify firewall rules. Shaping is optional and opt-in via CLI.
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
from typing import Dict, List, Tuple

import joblib
import pyshark
import pandas as pd


PORT_LABELS: Dict[int, str] = {5555: "VoIP", 6666: "FTP", 7777: "HTTP"}
_created_rules: List[str] = []  # Track created rules for cleanup


def cleanup_rules() -> None:
    """Remove firewall rules created during this session."""
    if not _created_rules:
        return
    print("\n[i] Cleaning up firewall rules created during this session...")
    for rule_name in _created_rules:
        try:
            subprocess.run(
                ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"],
                check=False,
                capture_output=True,
                text=True,
            )
            print(f"[i] Removed: {rule_name}")
        except Exception as e:
            print(f"[!] Failed to remove {rule_name}: {e}")


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
        # FIXED: Removed dst_port from features to prevent label leakage
        # Keep dst_port for display purposes only
        return {
            "protocol": protocol,
            "length": length,
            "src_port": src_port,
            "_dst_port_display": dst_port,  # For display only, not used in prediction
        }
    except Exception:
        return None


def block_port_windows(port: int, name: str, protocol: str, dry_run: bool = False) -> None:
    """Create an outbound and inbound block rule for the target port and protocol (TCP/UDP).
    
    Args:
        port: Port number to block
        name: Label name (VoIP/FTP/HTTP)
        protocol: Protocol (TCP/UDP)
        dry_run: If True, print what would be done without executing
    """
    global _created_rules
    for direction in ("in", "out"):
        rule_name = f"AI-Traffic-Shaper {name} {protocol} {direction} port {port}"
        if dry_run:
            print(f"[DRY-RUN] Would create firewall rule: {rule_name}")
            continue
        try:
            result = subprocess.run(
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
            if result.returncode == 0:
                _created_rules.append(rule_name)
                print(f"[+] Created firewall rule: {rule_name}")
            else:
                print(f"[!] Failed to create rule {rule_name}: {result.stderr}")
        except Exception as e:
            print(f"[x] Failed to add firewall rule: {e}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live prediction and optional shaping of traffic")
    p.add_argument("--model", default="traffic_model.pkl", help="Path to saved sklearn Pipeline")
    p.add_argument("--interface", "-i", required=True, help="Capture interface (index or name)")
    p.add_argument("--duration", "-d", type=int, default=15, help="Duration to run (seconds)")
    p.add_argument("--shape", action="store_true", help="Enable shaping (block predicted ports via Windows Firewall)")
    p.add_argument("--dry-run", action="store_true", help="Show what firewall rules would be created without actually creating them")
    p.add_argument("--no-cleanup", action="store_true", help="Don't automatically remove firewall rules on exit")
    p.add_argument(
        "--filter",
        default="tcp.port==6666 or tcp.port==7777 or udp.port==5555",
        help="Display filter",
    )
    args = p.parse_args(argv)

    # Safety checks
    if args.shape and not args.dry_run:
        print("⚠️  WARNING: Traffic shaping will modify Windows Firewall rules!")
        print("    - This can block legitimate traffic if predictions are wrong")
        print("    - Only use on isolated lab networks")
        print("    - Rules will be auto-removed on normal exit (use --no-cleanup to keep)")
        print("    - Use scripts/cleanup_firewall_rules.ps1 to manually remove rules")
        response = input("\nDo you want to continue? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("[!] Aborted by user")
            return 0

    # Register cleanup handler unless --no-cleanup is specified
    if args.shape and not args.no_cleanup and not args.dry_run:
        atexit.register(cleanup_rules)

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
            # Prepare a single-row prediction input (exclude display-only fields)
            dst_port_display = feats.pop("_dst_port_display", "N/A")
            X = pd.DataFrame([feats])
            pred = pipe.predict(X)[0]
            print(f"Predicted: {pred} for dst_port={dst_port_display}, protocol={feats['protocol']}")
            if args.shape:
                # Map predicted label to port (simple mapping) and associated protocol
                label_to_port = {v: k for k, v in PORT_LABELS.items()}
                port = label_to_port.get(str(pred))
                proto = "UDP" if str(pred) == "VoIP" else "TCP"
                key = (proto, port) if port else None
                if port and key not in blocked:
                    mode = "DRY-RUN" if args.dry_run else "APPLYING"
                    print(f"[i] {mode} firewall block for {pred} on {proto} port {port}")
                    block_port_windows(port, str(pred), proto, dry_run=args.dry_run)
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
