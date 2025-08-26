# PBL: AI Traffic Shaper — Network Analytics and ML Classification

## 1) Project summary
Build an end-to-end system that:
- Generates realistic traffic (VoIP-like UDP, FTP/HTTP-like TCP).
- Captures packets with TShark/PyShark and extracts features.
- Assembles a labeled dataset and trains an ML model to classify traffic.
- Optionally shapes traffic on Windows via firewall rules based on predictions.

Core files in this workspace:
- `traffic_generator.py`: synthetic traffic via scapy or sockets (Windows-friendly).
- `capture_features.py`: live capture → `dataset.csv` (label by dst port).
- `packet_capture/extract_features.py`: parse existing PCAP/PCAPNG into features.
- `train_model.py`: scikit-learn Pipeline; saves `traffic_model.pkl`.
- `batch_predict.py`: offline evaluation on CSVs.
- `predict_and_shape.py`: live predictions, optional Windows firewall shaping.
- Data artifacts: `captured_traffic.pcapng`, `dataset.csv`, `traffic_features.csv`.

Deliverable: a working demo plus a short report (design, results, lessons).

## 2) Learning outcomes
Students will be able to:
- Explain packet anatomy, ports/protocols, and capture filtering.
- Operate TShark/PyShark for live capture with controlled stopping.
- Engineer packet-level features and create labeled datasets.
- Build and evaluate a scikit-learn Pipeline (preprocessing + classifier).
- Reason about model generalization, drift, and class imbalance.
- Implement cautious, reversible “shaping” on Windows via `netsh advfirewall`.

## 3) Audience and prerequisites
- Audience: Intermediate networking or ML students.
- Prereqs: Basic Python, OSI/TCP-IP basics, virtualenv know-how, Windows admin familiarity. Wireshark/TShark + Npcap installed.

## 4) Timeline (suggested)
- Week 1: Networking fundamentals, toolchain setup, baseline capture.
- Week 2: Dataset creation (generator + capture), model training + evaluation.
- Week 3: Live prediction demo + optional shaping, reflections and improvements.

## 5) Scenario and problem statement
ISPs and enterprise networks often need to recognize flows (e.g., VoIP vs. web) to prioritize or control traffic. The task: learn to classify traffic types in real time using packet-derived features, then optionally act on predictions to shape traffic, while acknowledging safety and ethics constraints.

## 6) System overview and architecture
Flow (data path):
Traffic Generator → Network Stack → Adapter → TShark/PyShark Capture → Feature Extraction → `dataset.csv` → Training (Pipeline) → `traffic_model.pkl` → Live Capture Features → Model Inference → Optional Windows Firewall Rules.

Key choices:
- Windows-friendly socket generator for reliable visibility.
- PyShark with TShark `-a duration` to guarantee capture stop.
- Port-to-label mapping: 5555 (VoIP-like UDP), 6666 (FTP-like TCP), 7777 (HTTP-like TCP), else Other.

## 7) Networking fundamentals in play
- Packet anatomy: L2 (MAC), L3 (IP src/dst), L4 (TCP/UDP src/dst ports), payload length.
- Protocols and ports:
  - UDP (connectionless; used here for “VoIP-like” bursts).
  - TCP (connection-oriented; used for FTP/HTTP-like packets).
- Capture drivers on Windows:
  - Npcap provides capture APIs. Loopback adapter often works out-of-the-box; Wi‑Fi monitoring may require specific install options.
- Filters:
  - BPF filter (kernel-level select, e.g., `udp or tcp`).
  - Display filter (post-capture select, Wireshark syntax).
- Practicalities:
  - Loopback captures are reliable for this lab.
  - Wi‑Fi capture can show 0 packets if driver or Npcap mode is limited.
- Safety and privacy: capture only your own lab traffic; never capture sensitive or unauthorized networks.

## 8) AI/ML fundamentals in play
- Features (single-packet baseline): length, src_port, dst_port (numeric), protocol (categorical).
- Labels: mapped from destination ports by convention (VoIP/FTP/HTTP/Other).
- Pipeline:
  - ColumnTransformer: StandardScaler for numeric, OneHotEncoder for protocol.
  - Estimator: RandomForestClassifier (robust, interpretable, low-tuning).
- Evaluation: train/val split, confusion matrix, classification report, accuracy.
- Risks:
  - Overfitting to synthetic traffic; domain drift on real networks.
  - Label leakage via ports; mitigate by adding richer features later (flow stats, timing).
  - Class imbalance and small-sample variance.

## 9) Tools and environment
- Windows 10/11; PowerShell admin for firewall shaping.
- Wireshark/TShark and Npcap (WinPcap-compatible mode with loopback support).
- Python 3.13 venv in `traffic_env/`.
- Core libs: pyshark, scapy (optional), pandas, scikit-learn, numpy, joblib.

## 10) Activities (step-by-step)

A. Setup and sanity checks
- Verify TShark available (version prints).
- List interfaces and identify Loopback and Wi‑Fi indices.
- Confirm virtualenv activation (`traffic_env\\Scripts\\Activate.ps1`).

B. Capture baseline traffic (loopback first)
- Run `capture_features.py` with loopback interface and `--duration` (auto-stop).
- Use `--no-filter` once to confirm packets appear; then add filters to reduce noise.
- Inspect `dataset.csv` rows (protocol, length, src/dst IP/port, label).

C. Generate labeled synthetic traffic
- Use `traffic_generator.py --method socket` to send:
  - VoIP-like UDP bursts to 127.0.0.1:5555.
  - TCP-like traffic to 127.0.0.1:6666 (FTP-like), 7777 (HTTP-like).
- Simultaneously capture with `capture_features.py` on loopback; confirm label coverage.

D. Train and evaluate the model
- Train with `train_model.py` on the combined `dataset.csv` (optionally include earlier `traffic_features.csv`).
- Check confusion matrix and report; if “Other” dominates, consider `--keep-other` or filter it.
- Run `batch_predict.py` to evaluate on a held-out CSV.

E. Live prediction and optional shaping
- Run `predict_and_shape.py` on loopback. Observe predicted labels in real time.
- Optionally enable shaping: blocks by predicted label’s protocol/port, de-duplicates rules.
- Validate add/remove rules carefully; keep a rollback plan.

F. Reflection and iteration
- Document what improved accuracy most (features, class balance).
- Identify gaps (e.g., Wi‑Fi capture issues, need for flow-level features).
- Propose next iteration plan (extensions below).

## 11) Assessment rubric (holistic)
- Networking proficiency (20%): correct capture setup, filters, interface selection, safety.
- Data/feature quality (20%): labeling fidelity, balanced coverage, rationale for features.
- ML pipeline and evaluation (30%): clean preprocessing, justified model, sound metrics and interpretation.
- System integration (20%): working end-to-end demo; predictable, reversible shaping.
- Reporting and reflection (10%): clarity, limitations, ethical considerations, next steps.

Levels: Exemplary (meets all with evidence), Proficient (minor gaps), Developing (partial integration or weak evaluation), Beginning (setup only).

## 12) Troubleshooting guide
- “0 packets captured” on Wi‑Fi:
  - Reinstall Npcap with WinPcap-compatible mode; enable loopback and raw 802.11 if available.
  - Run TShark as admin; try Ethernet or Loopback to isolate driver issues.
  - Test direct TShark capture with small duration limit.
- Capture never stops:
  - Ensure `-a duration:n` is set; avoid only timeout-based sniffing.
- No packets from generator:
  - Prefer `--method socket` on Windows; scapy may not reflect in capture stacks.
  - Confirm matching ports (5555/6666/7777) and loopback destination.
- Firewall shaping:
  - Requires admin; ensure rule names unique; verify protocol (UDP for VoIP).
  - Keep a cleanup step to remove rules.
- Model accuracy too high to be true:
  - Check for label leakage (e.g., dst_port dominating). Add ablations or more features; validate on different traffic sources.

## 13) Extensions and stretch goals
- Flow-level features: sliding-window stats (pkts/sec, inter-arrival, byte rate).
- Protocol parsing: DNS/TLS SNI, HTTP headers (PyShark fields).
- Cross-platform capture (Linux/macOS) and containerized lab.
- Real QoS integration (DSCP tagging) instead of blocking.
- Online learning or drift detection; more realistic traffic mixes.
- Add tests and CI: smoke tests for each script and linters.

## 14) Ethics, safety, and scope
- Limit capture to lab interfaces and your own traffic.
- Obtain consent; avoid storing payloads; keep only metadata needed.
- Shaping/blocking can disrupt legitimate traffic—use on isolated lab networks and always with rollback.

## 15) Glossary (selected)
- BPF: Berkeley Packet Filter, kernel-level capture filter.
- Display filter: Wireshark post-capture filter.
- Loopback adapter: Virtual NIC for local host traffic.
- ColumnTransformer: scikit-learn construct to preprocess different column types jointly.
- RandomForest: Ensemble of decision trees; robust baseline classifier.
