"""
simulate_real_traffic.py

Simulate realistic network traffic patterns on different ports
to test the model against "real-world-like" data.
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

def generate_realistic_samples(n_samples=500):
    """Generate realistic traffic that mimics real-world patterns."""
    samples = []
    timestamp = datetime.now().timestamp()
    
    # Real-world traffic characteristics (more diverse than synthetic)
    for i in range(n_samples):
        # Random realistic scenarios
        scenario = np.random.choice(['web_browsing', 'video_stream', 'file_download', 
                                      'voip_call', 'gaming', 'email'], 
                                     p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10])
        
        if scenario == 'web_browsing':
            # HTTP/HTTPS traffic (ports 80, 443, 8080)
            protocol = 'TCP'
            dst_port = np.random.choice([80, 443, 8080], p=[0.2, 0.7, 0.1])
            src_port = np.random.randint(49152, 65535)  # Ephemeral ports
            # Mix of requests (small) and responses (larger)
            if np.random.random() < 0.3:
                length = np.random.randint(100, 400)  # Request
            else:
                length = np.random.randint(500, 1460)  # Response (MTU-limited)
            label = 'HTTP'  # Real HTTP traffic
            
        elif scenario == 'video_stream':
            # Video streaming (YouTube, Netflix style)
            protocol = 'TCP'
            dst_port = np.random.choice([443, 80])
            src_port = np.random.randint(49152, 65535)
            length = np.random.randint(1400, 1500)  # Large packets
            label = 'HTTP'  # Classified as HTTP
            
        elif scenario == 'file_download':
            # FTP or large file transfers
            protocol = 'TCP'
            dst_port = np.random.choice([21, 20, 443])  # FTP or HTTPS
            src_port = np.random.randint(40000, 50000)
            length = np.random.randint(1200, 1500)  # Large packets
            label = 'FTP'  # Classified as FTP-like
            
        elif scenario == 'voip_call':
            # VoIP (Skype, Zoom, etc.)
            protocol = 'UDP'
            dst_port = np.random.choice([3478, 3479, 5004])  # STUN/RTP ports
            src_port = np.random.randint(50000, 60000)
            length = np.random.randint(150, 250)  # Small voice packets
            label = 'VoIP'
            
        elif scenario == 'gaming':
            # Online gaming
            protocol = 'UDP'
            dst_port = np.random.randint(20000, 30000)
            src_port = np.random.randint(50000, 60000)
            length = np.random.randint(60, 200)  # Small, frequent
            label = 'Other'
            
        elif scenario == 'email':
            # Email (SMTP, IMAP)
            protocol = 'TCP'
            dst_port = np.random.choice([25, 587, 993, 143])
            src_port = np.random.randint(49152, 65535)
            length = np.random.randint(200, 800)
            label = 'Other'
        
        samples.append({
            'timestamp': timestamp + i * 0.01,
            'protocol': protocol,
            'length': length,
            'src_ip': f'192.168.1.{np.random.randint(2, 254)}',
            'dst_ip': f'203.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}',
            'src_port': src_port,
            'dst_port': dst_port,
            'label': label
        })
    
    return samples

def main():
    print("[+] Generating realistic network traffic samples...")
    print("    (simulating web browsing, video streaming, file downloads, VoIP, etc.)\n")
    
    # Generate realistic samples
    samples = generate_realistic_samples(n_samples=600)
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Save to CSV
    output_file = "real_traffic.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created {output_file} with {len(df)} realistic samples\n")
    print("📊 Label Distribution:")
    print(df['label'].value_counts())
    print("\n📊 Protocol Distribution:")
    print(df['protocol'].value_counts())
    print("\n📊 Common Destination Ports:")
    print(df['dst_port'].value_counts().head(10))
    print("\n📊 Sample data:")
    print(df.head(10).to_string())
    
    return 0

if __name__ == "__main__":
    exit(main())
