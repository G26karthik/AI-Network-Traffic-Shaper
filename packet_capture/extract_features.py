import os
import argparse
import pyshark
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# UPDATED: Align with main pipeline labeling (VoIP/FTP/HTTP on ports 5555/6666/7777)
# This function now provides BOTH the main pipeline labels AND real-world labels
def get_label(src_port, dst_port, use_synthetic_labels=True):
    """
    Label packets based on port numbers.
    
    Args:
        src_port: Source port
        dst_port: Destination port
        use_synthetic_labels: If True, use synthetic lab labels (VoIP/FTP/HTTP).
                              If False, use real-world labels (Web/DNS/SSH/etc.)
    """
    try:
        src = int(src_port) if src_port != 'N/A' else None
        dst = int(dst_port) if dst_port != 'N/A' else None
    except:
        return 'Unknown'

    ports = [src, dst]
    
    if use_synthetic_labels:
        # Synthetic lab labels matching main pipeline (for dataset.csv)
        if any(p == 5555 for p in ports):
            return 'VoIP'
        elif any(p == 6666 for p in ports):
            return 'FTP'
        elif any(p == 7777 for p in ports):
            return 'HTTP'
        else:
            return 'Other'
    else:
        # Real-world labels (for actual captured traffic)
        if any(p in [80, 443, 8080] for p in ports):
            return 'Web'
        elif any(p == 53 for p in ports):
            return 'DNS'
        elif any(p in [20, 21] for p in ports):
            return 'FTP'
        elif any(p == 22 for p in ports):
            return 'SSH'
        elif any(p in [25, 587] for p in ports):
            return 'Email'
        elif any(p == 1900 for p in ports):
            return 'SSDP'
        elif any(p == 5353 for p in ports):
            return 'mDNS'
        elif any(p == 5355 for p in ports):
            return 'LLMNR'
        elif any(p == 137 for p in ports):
            return 'NetBIOS'
        else:
            return 'Other'

def _resolve_pcap_path(pcap_file: str) -> str:
    if os.path.isabs(pcap_file) and os.path.isfile(pcap_file):
        return pcap_file

    candidates = []
    cwd = os.getcwd()
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(script_dir)

    candidates.append(os.path.join(cwd, pcap_file))
    candidates.append(os.path.join(script_dir, pcap_file))
    candidates.append(os.path.join(parent_dir, pcap_file))

    for c in candidates:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError("PCAP not found. Tried: " + "; ".join(candidates))


def extract_features(pcap_file='captured_traffic.pcapng', raw_csv='traffic_features.csv', ml_csv='traffic_features_ml.csv', use_synthetic_labels=True):
    """
    Extract features from a PCAP file and save to CSV.
    
    Args:
        pcap_file: Path to PCAP/PCAPNG file
        raw_csv: Output path for raw features
        ml_csv: Output path for ML-ready features
        use_synthetic_labels: If True, use synthetic lab labels (VoIP/FTP/HTTP).
                             If False, use real-world labels (Web/DNS/SSH/etc.)
    """
    resolved_pcap = _resolve_pcap_path(pcap_file)
    print(f"[+] Reading pcap: {resolved_pcap}")
    cap = pyshark.FileCapture(resolved_pcap)

    extracted_data = []

    for pkt in cap:
        try:
            timestamp = float(pkt.sniff_timestamp)
            protocol = pkt.highest_layer
            length = int(pkt.length)

            src_ip = pkt.ip.src if 'IP' in pkt else 'N/A'
            dst_ip = pkt.ip.dst if 'IP' in pkt else 'N/A'
            src_port = pkt[pkt.transport_layer].srcport if hasattr(pkt, 'transport_layer') else 'N/A'
            dst_port = pkt[pkt.transport_layer].dstport if hasattr(pkt, 'transport_layer') else 'N/A'

            # Use the specified labeling scheme
            label = get_label(src_port, dst_port, use_synthetic_labels)

            extracted_data.append({
                'timestamp': timestamp,
                'protocol': protocol,
                'length': length,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'label': label
            })

        except Exception:
            continue

    # ✅ Save raw dataset
    df = pd.DataFrame(extracted_data)
    df.to_csv(raw_csv, index=False)
    print(f"[+] Extracted {len(df)} packets. Raw data saved to {raw_csv}")

    # ✅ Preprocess for ML
    print("[+] Preprocessing for ML...")
    ml_df = df.copy()

    # Drop IPs (optional) because they are high-cardinality categorical
    ml_df.drop(['src_ip', 'dst_ip'], axis=1, inplace=True)

    # Encode categorical features
    for col in ['protocol', 'label']:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])

    # Convert ports to numeric (handle N/A)
    ml_df['src_port'] = pd.to_numeric(ml_df['src_port'], errors='coerce').fillna(0)
    ml_df['dst_port'] = pd.to_numeric(ml_df['dst_port'], errors='coerce').fillna(0)

    # Normalize numeric columns
    scaler = MinMaxScaler()
    ml_df[['timestamp', 'length', 'src_port', 'dst_port']] = scaler.fit_transform(
        ml_df[['timestamp', 'length', 'src_port', 'dst_port']]
    )

    # Save ML-ready CSV
    ml_df.to_csv(ml_csv, index=False)
    print(f"[+] ML-ready dataset saved to {ml_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract simple features from a pcap using pyshark")
    parser.add_argument("--pcap", default="captured_traffic.pcapng", help="Path to pcap file (absolute or relative)")
    parser.add_argument("--raw", default="traffic_features.csv", help="Output raw CSV file path")
    parser.add_argument("--ml", default="traffic_features_ml.csv", help="Output ML-ready CSV file path")
    parser.add_argument("--synthetic-labels", action="store_true", default=True, 
                       help="Use synthetic lab labels (VoIP/FTP/HTTP on ports 5555/6666/7777). Default: True")
    parser.add_argument("--real-labels", dest="synthetic_labels", action="store_false",
                       help="Use real-world labels (Web/DNS/SSH/etc. on standard ports)")
    args = parser.parse_args()

    extract_features(pcap_file=args.pcap, raw_csv=args.raw, ml_csv=args.ml, use_synthetic_labels=args.synthetic_labels)
