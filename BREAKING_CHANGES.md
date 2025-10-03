# ⚠️ IMPORTANT: Recent Breaking Changes (2025-10-03)

## Critical Fix: Label Leakage Resolved

We've fixed a **critical issue** where the model was using `dst_port` as a feature, causing severe label leakage. This made the model achieve unrealistically high accuracy (~98%) on synthetic data but fail on real traffic.

### What This Means for You:

1. **Existing trained models will NOT work** with updated code
2. **You MUST retrain** after pulling the latest changes
3. **Expected accuracy will be lower** (60-85% instead of 98%) - **this is correct!**

### Quick Migration:

```powershell
# 1. Delete old model
Remove-Item traffic_model.pkl -ErrorAction SilentlyContinue

# 2. Retrain
.\traffic_env\Scripts\python.exe .\train_model.py --data dataset.csv

# 3. Verify
.\traffic_env\Scripts\python.exe .\batch_predict.py --model traffic_model.pkl --data dataset.csv
```

📖 **See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for complete instructions**

---

## Other Improvements

- ✅ Added traffic shaping safety features (dry-run mode, auto-cleanup)
- ✅ Fixed labeling consistency across all scripts
- ✅ Clarified PyTorch and optional dependencies
- ✅ Removed duplicate/obsolete files

📋 **See [CHANGELOG.md](CHANGELOG.md) for full details**

---

🔄 **Already migrated?** You can delete this file after reading.
