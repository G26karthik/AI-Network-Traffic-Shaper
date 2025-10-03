# Changelog

## [Unreleased] - 2025-10-03

### 🔧 Fixed - Critical Issues
- **FIXED: Label Leakage** - Removed `dst_port` from features to prevent model from learning port numbers instead of traffic patterns
  - Updated `train_model.py`, `batch_predict.py`, `predict_and_shape.py`
  - Updated deep learning modules (`deep/data.py`, `deep/train_torch.py`, `serve_api.py`)
  - Models now use only `protocol`, `length`, and `src_port` as features
  - ⚠️ **BREAKING**: Existing trained models (`traffic_model.pkl`) must be retrained with new feature set

### ✨ Improved
- **Labeling Consistency** - `packet_capture/extract_features.py` now supports both synthetic and real-world labels
  - `--synthetic-labels` (default): VoIP/FTP/HTTP on ports 5555/6666/7777 (matches main pipeline)
  - `--real-labels`: Web/DNS/SSH/Email/etc. on standard ports (for real traffic analysis)
  
- **Traffic Shaping Safety** - Enhanced `predict_and_shape.py` with safety features:
  - Added `--dry-run` flag to preview firewall rules without creating them
  - Interactive confirmation prompt before modifying firewall
  - Automatic cleanup of firewall rules on exit (disable with `--no-cleanup`)
  - Comprehensive warnings about risks and rollback procedures

- **Dependency Management** - Clarified optional dependencies in `requirements.txt`:
  - Core dependencies clearly marked
  - PyTorch installation instructions added
  - Commented out unused features (Kafka, Spark)

### 🗑️ Removed
- Deleted duplicate `traffic_simulation/traffic_generator.py` (use main `traffic_generator.py` instead)
- Removed obsolete pickle files: `label_encoder.pkl`, `scaler.pkl`, `network_traffic_model.pkl`

### 📚 Documentation
- Added this CHANGELOG.md
- Created MIGRATION_GUIDE.md with upgrade instructions
- Will update README.md with warnings about label leakage fix

### ⚠️ Breaking Changes
- **Feature set changed**: Models trained before this update will NOT work with new code
- **Retraining required**: Run `train_model.py` again on your dataset after updating
- **API change**: `serve_api.py` TrafficFeatures model no longer includes `dst_port` field

### 🔄 Migration Path
1. Pull latest changes
2. Delete old `traffic_model.pkl` (if it exists)
3. Recapture traffic or use existing `dataset.csv`
4. Retrain model: `.\traffic_env\Scripts\python.exe .\train_model.py --data dataset.csv`
5. Update any custom scripts that reference `dst_port` feature

---

## Previous versions
No formal versioning before this point. See git history for earlier changes.
