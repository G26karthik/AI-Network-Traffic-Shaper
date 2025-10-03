# AI-Based Network Traffic Shaper (Windows)

> ⚠️ **IMPORTANT**: Recent update fixed critical label leakage issue. **Existing models must be retrained!**  
> See [BREAKING_CHANGES.md](BREAKING_CHANGES.md) and [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details.

Generate synthetic VoIP/FTP/HTTP traffic, capture with PyShark/TShark, extract features, train a model, and optionally shape traffic via Windows Firewall rules.

---

## Features

- Traffic generation via scapy or Windows socket APIs (VoIP-like UDP, FTP/HTTP-like TCP)
- Live capture using PyShark with TShark auto-stop and filter controls
- **Fixed**: Proper feature engineering (removed label leakage)
- Training pipeline (ColumnTransformer + RandomForest) saved as a single sklearn Pipeline
- Optional PyTorch deep learning models (MLP, GRU)
- Batch evaluation on CSV and live prediction from capture
- **Enhanced**: Safe traffic shaping with dry-run mode and automatic cleanup

---

## Recent Changes (2025-10-03)

### 🔧 Critical Fixes
- **Label Leakage Fixed**: Removed `dst_port` from features to prevent the model from cheating
  - **BREAKING**: Existing models must be retrained
  - Expected accuracy: 60-85% (realistic) instead of 98% (artificial)
  
### ✨ Improvements
- Traffic shaping now has safety features (`--dry-run`, auto-cleanup, warnings)
- Labeling consistency across all scripts
- Better dependency documentation

📖 **Full details**: [CHANGELOG.md](CHANGELOG.md) | **Migration**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## Setup

Prerequisites on Windows:
- Wireshark/TShark installed (ensure `tshark -v` works in a new PowerShell)
- Npcap installed (enable WinPcap-compatible mode; loopback support recommended)
- Administrator PowerShell for capture and shaping steps

### Installation

```powershell
# Create virtual environment
python -m venv traffic_env
./traffic_env/Scripts/Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Optional: Install PyTorch for deep learning models
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Verify TShark
tshark -v
```

### Quick Start (After Installation)

```powershell
# 1. Generate traffic and capture (run as Administrator)
./traffic_env/Scripts/python.exe ./capture_features.py --interface 8 --duration 15 --output dataset.csv
# In another terminal:
./traffic_env/Scripts/python.exe ./traffic_generator.py --type all --duration 15 --pps 30 --dst 127.0.0.1

# 2. Train model
./traffic_env/Scripts/python.exe ./train_model.py --data dataset.csv --model-out traffic_model.pkl

# 3. Evaluate
./traffic_env/Scripts/python.exe ./batch_predict.py --model traffic_model.pkl --data dataset.csv
```

---

## Usage

The repository provides modular scripts and a one-shot pipeline.

### 1) List capture interfaces

```powershell
./traffic_env/Scripts/python.exe ./capture_features.py --list
```

### 2) Generate synthetic traffic

Socket method (recommended on Windows loopback):

```powershell
./traffic_env/Scripts/python.exe ./traffic_generator.py --type all --duration 15 --pps 30 --dst 127.0.0.1 --method socket
```

Scapy method (may require Administrator):

```powershell
./traffic_env/Scripts/python.exe ./traffic_generator.py --type voip --duration 10 --pps 50 --method scapy
```

CLI options for `traffic_generator.py`:
- `--type {voip,ftp,http,all}`: which traffic to emit (default: all)
- `--dst <ip>`: destination IP (default: 127.0.0.1)
- `--duration <seconds>`: how long to run (default: 10)
- `--pps <rate>`: target packets/sec (default: 30)
- `--voip-port|--ftp-port|--http-port`: override default ports 5555/6666/7777
- `--method {scapy,socket}`: packet send method (Windows default: socket)

### 3) Capture and build a dataset

```powershell
./traffic_env/Scripts/python.exe ./capture_features.py --interface 1 --duration 15 --output dataset.csv
```

Useful flags:
- `--filter`: Wireshark display filter (default targets ports 5555/6666/7777)
- `--bpf`: capture filter (kernel-level)
- `--no-filter`: disable both filters for debugging

Output columns: `timestamp, protocol, length, src_ip, dst_ip, src_port, dst_port, label`

### 4) Train a model

```powershell
./traffic_env/Scripts/python.exe ./train_model.py --data dataset.csv --model-out traffic_model.pkl
```

Options:
- `--keep-other`: include the `Other` label in training (default filters it out)
- `--test-size`, `--random-state`: control the train/test split

### 5) Batch evaluate on a CSV

```powershell
./traffic_env/Scripts/python.exe ./batch_predict.py --model traffic_model.pkl --data dataset.csv
```

### 6) Live prediction (optional shaping)

⚠️ **WARNING**: Traffic shaping modifies Windows Firewall and can block legitimate traffic!

Without shaping (safe):

```powershell
./traffic_env/Scripts/python.exe ./predict_and_shape.py --model traffic_model.pkl --interface 1 --duration 15
```

With dry-run mode (preview only, no changes):

```powershell
./traffic_env/Scripts/python.exe ./predict_and_shape.py --model traffic_model.pkl --interface 1 --duration 15 --shape --dry-run
```

With actual firewall shaping (requires Administrator, auto-cleanup on exit):

```powershell
./traffic_env/Scripts/python.exe ./predict_and_shape.py --model traffic_model.pkl --interface 1 --duration 15 --shape
```

Manual cleanup if needed:

```powershell
./scripts/cleanup_firewall_rules.ps1
```

Shaping maps predicted labels to ports/protocols:
- VoIP → UDP/5555
- FTP → TCP/6666
- HTTP → TCP/7777

### 7) End-to-end pipeline (capture → generate → train → evaluate)

```powershell
./traffic_env/Scripts/python.exe ./run_pipeline.py --duration 12 --pps 30 --dst 127.0.0.1 --fresh
```

Run options: `--interface`, `--duration`, `--pps`, `--dst`, `--dataset`, `--fresh`, `--model-out`, `--keep-other`.

---

## Important Notes and Limitations

### Model Limitations
- **Features used**: `protocol`, `length`, `src_port` (dst_port deliberately excluded to prevent label leakage)
- **Expected accuracy**: 60-85% on synthetic data (realistic, not artificially inflated)
- **Generalization**: Synthetic traffic patterns differ from real-world traffic
- **Port-based training**: Model trained on specific ports (5555/6666/7777) may not generalize to standard ports

### Recommendations for Production Use
This is an **educational project**. For production traffic classification:
1. Use real labeled datasets (not synthetic)
2. Add flow-level features (packet rate, inter-arrival time, byte distributions)
3. Consider deep packet inspection (DPI) for protocol-specific features
4. Implement proper QoS (Quality of Service) instead of port blocking
5. Use streaming architectures for real-time processing at scale

### Safety Considerations
- **Capture only on lab networks** - never on production or unauthorized networks
- **Traffic shaping is destructive** - test with `--dry-run` first
- **Firewall rules persist** - use cleanup scripts or `--no-cleanup` flag appropriately
- **Admin rights required** - capture and shaping need elevated privileges

---

## Folder Structure

```
AI-Traffic-Shaper/
├─ batch_predict.py                 # Batch evaluate a saved model on a CSV
├─ capture_features.py              # Live capture → features → dataset.csv
├─ packet_capture/
│  ├─ extract_features.py           # Parse a PCAP into CSV(s)
│  └─ capture_with_pyshark.py       # Minimal capture to pcapng
├─ predict_and_shape.py             # Live predict; optional Windows Firewall shaping
├─ run_pipeline.py                  # One-shot end-to-end runner
├─ traffic_generator.py             # Synthetic traffic generator (socket/scapy)
├─ traffic_simulation/
│  └─ traffic_generator.py          # Alternate location of generator (duplicate logic)
├─ train_model.py                   # Train sklearn Pipeline and save model
├─ requirements.txt                 # Python dependencies
├─ dataset.csv                      # Example dataset (generated)
├─ traffic_model.pkl                # Example trained model artifact
├─ traffic_features.csv             # PCAP-extracted CSV (optional)
├─ docs/
│  ├─ PBL_AI_Traffic_Shaper.md      # Student-facing PBL guide
│  └─ Instructor_Notes.md           # Instructor-only notes
└─ traffic_env/                     # Local Python virtual environment (Windows)
```

---

## APIs

This repository exposes CLI “APIs” via Python entry points rather than HTTP services. See the Usage section for command-line options of each module:
- `traffic_generator.py` — generate traffic
- `capture_features.py` — capture and write dataset entries
- `train_model.py` — train and persist a model
- `batch_predict.py` — offline evaluation
- `predict_and_shape.py` — live prediction and optional shaping
- `packet_capture/extract_features.py` — offline PCAP → CSV
- `run_pipeline.py` — orchestrate all stages

---

## Contributing

Contributions are welcome! To propose changes:
1. Open an issue describing the improvement or bug.
2. Fork the repo and create a feature branch.
3. Use a virtual environment and ensure scripts run without errors on Windows PowerShell.
4. If you change script behavior, update this README and `docs/` accordingly.
5. Submit a pull request with a clear description and screenshots/logs when relevant.

Notes:
- There are currently no automated tests; consider adding smoke tests for each script.
- Please keep Windows support in mind (PowerShell examples, TShark/Npcap).

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Badges and Highlights (suggested)

Add these once corresponding metadata exists in your repo:

```markdown
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-Add%20LICENSE-important)
```

If you adopt GitHub Actions, you can add a build badge, for example:

```markdown
![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)
```

---

## Acknowledgements

- Built with PyShark/TShark on Npcap for Windows capture
- Scikit-learn Pipeline for simple, reproducible ML training

For a project-based learning guide and instructor notes, see `docs/`.

---

## Maintenance tips

- Remove Windows Firewall rules created by live shaping if needed:

```powershell
./scripts/cleanup_firewall_rules.ps1
```

