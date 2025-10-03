"""
create_balanced_dataset.py

Create a balanced synthetic dataset for training when actual traffic capture
doesn't capture all types (e.g., TCP connections to non-existent servers).
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_traffic_samples(traffic_type, dst_port, protocol, n_samples=300):
    """Generate synthetic traffic samples for a given type."""
    samples = []
    
    for i in range(n_samples):
        if traffic_type == "VoIP":
            # VoIP: Small UDP packets (RTP-like)
            length = np.random.randint(160, 200)
            proto = "UDP"
            src_port = np.random.randint(50000, 60000)
        elif traffic_type == "FTP":
            # FTP: Variable TCP packets (control + data)
            length = np.random.choice([
                np.random.randint(40, 100),    # Control messages (30%)
                np.random.randint(500, 1500)   # Data packets (70%)
            ], p=[0.3, 0.7])
            proto = "TCP"
            src_port = np.random.randint(40000, 50000)
        elif traffic_type == "HTTP":
            # HTTP: Medium TCP packets (requests/responses)
            length = np.random.choice([
                np.random.randint(200, 400),   # Requests (40%)
                np.random.randint(500, 1400)   # Responses (60%)
            ], p=[0.4, 0.6])
            proto = "TCP"
            src_port = np.random.randint(35000, 45000)
        
        samples.append({
            'timestamp': datetime.now().timestamp() + i * 0.01,
            'protocol': proto,
            'length': length,
            'src_ip': '127.0.0.1',
            'dst_ip': '127.0.0.1',
            'src_port': src_port,
            'dst_port': dst_port,
            'label': traffic_type
        })
    
    return samples

def main():
    print("[+] Creating balanced synthetic dataset...")
    
    # Generate samples for each traffic type
    voip_samples = generate_traffic_samples("VoIP", 5555, "UDP", n_samples=400)
    ftp_samples = generate_traffic_samples("FTP", 6666, "TCP", n_samples=400)
    http_samples = generate_traffic_samples("HTTP", 7777, "TCP", n_samples=400)
    
    # Combine all samples
    all_samples = voip_samples + ftp_samples + http_samples
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    output_file = "dataset_balanced.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Created {output_file} with {len(df)} samples")
    print("\n📊 Label Distribution:")
    print(df['label'].value_counts())
    print("\n📊 Protocol Distribution:")
    print(df['protocol'].value_counts())
    print("\n📊 Sample data:")
    print(df.head(10))
    
    return 0

if __name__ == "__main__":
    exit(main())
