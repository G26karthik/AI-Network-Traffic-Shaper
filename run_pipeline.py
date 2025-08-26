"""
run_pipeline.py

End-to-end pipeline: capture -> generate -> train -> evaluate.
Defaults target localhost on Npcap Loopback so it works even when on hotspot.

Steps:
1) Start capture_features.py for N seconds on selected interface writing to dataset.csv.
2) Start traffic_generator.py concurrently to produce VoIP/FTP/HTTP traffic to 127.0.0.1.
3) Train model on dataset.csv and save traffic_model.pkl.
4) Batch evaluate predictions on dataset.csv and print report.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional


def run(cmd: List[str], check: bool = True) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def popen(cmd: List[str]) -> subprocess.Popen:
    print("$", " ".join(cmd))
    return subprocess.Popen(cmd)


def tshark_list() -> List[str]:
    try:
        out = subprocess.check_output(["tshark", "-D"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print("[x] tshark -D failed. Is TShark in PATH?", e)
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def auto_interface() -> Optional[str]:
    lines = tshark_list()
    # Prefer Loopback for localhost synthetic traffic
    preferred = None
    for ln in lines:
        if "Loopback" in ln:
            preferred = ln
            break
    if not preferred:
        for name in ("Wi-Fi", "WiFi", "Ethernet"):
            for ln in lines:
                if name in ln:
                    preferred = ln
                    break
            if preferred:
                break
    if not preferred:
        return None
    # Extract numeric index if present (e.g., '8. \Device...')
    if "." in preferred:
        idx = preferred.split(".", 1)[0].strip()
        return idx if idx.isdigit() else preferred
    return preferred


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run end-to-end pipeline (capture -> generate -> train -> eval)")
    p.add_argument("--interface", "-i", default=None, help="Interface index/name (default: auto)")
    p.add_argument("--duration", "-d", type=int, default=12, help="Duration (seconds) for capture/generate")
    p.add_argument("--pps", type=int, default=30, help="Packets per second for generator")
    p.add_argument("--dst", default="127.0.0.1", help="Destination IP (default 127.0.0.1)")
    p.add_argument("--dataset", default="dataset.csv", help="Dataset CSV path (default dataset.csv)")
    p.add_argument("--fresh", action="store_true", help="Start with a fresh dataset (delete if exists)")
    p.add_argument("--model-out", default="traffic_model.pkl", help="Model output path")
    p.add_argument("--keep-other", action="store_true", help="Keep 'Other' class in training")
    args = p.parse_args(argv)

    py = sys.executable  # use the interpreter used to run this script (ideally venv)

    # Preflight: tshark
    try:
        ver = subprocess.check_output(["tshark", "-v"], text=True)
        print("[i]", ver.splitlines()[0])
    except Exception as e:
        print("[x] tshark not available:", e)
        return 2

    # Resolve interface
    iface = args.interface or auto_interface()
    if not iface:
        print("[x] Could not auto-detect interface. Use --interface after running 'tshark -D'.")
        return 2
    print(f"[i] Using interface: {iface}")

    # Fresh dataset option
    if args.fresh and os.path.exists(args.dataset):
        try:
            os.remove(args.dataset)
            print(f"[i] Removed existing {args.dataset}")
        except Exception as e:
            print(f"[!] Could not remove {args.dataset}: {e}")

    # Start capture
    cap_cmd = [py, "capture_features.py", "--interface", str(iface), "--duration", str(args.duration), "--output", args.dataset]
    cap_proc = popen(cap_cmd)

    # Give capture a head-start
    time.sleep(1.0)

    # Start generator
    gen_cmd = [py, "traffic_generator.py", "--type", "all", "--dst", args.dst, "--duration", str(args.duration), "--pps", str(args.pps)]
    gen_rc = run(gen_cmd, check=False)

    # Wait for capture to end (sniff has timeout)
    cap_proc.wait()

    # Train
    train_cmd = [py, "train_model.py", "--data", args.dataset, "--model-out", args.model_out]
    if args.keep_other:
        train_cmd.append("--keep-other")
    run(train_cmd, check=True)

    # Evaluate (batch)
    eval_cmd = [py, "batch_predict.py", "--model", args.model_out, "--data", args.dataset]
    run(eval_cmd, check=False)

    print("[✓] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
