# Migration Guide: Updating to Fixed Version

## Overview
This guide helps you migrate from the original version to the updated version that fixes critical label leakage issues and improves safety features.

---

## ⚠️ Critical: Label Leakage Fix

### What Changed
**Problem**: The original model used `dst_port` as a feature, which directly leaks the label (since labels are assigned by destination port). This caused the model to achieve artificially high accuracy on synthetic data but perform poorly on real traffic.

**Solution**: We removed `dst_port` from features. The model now learns from:
- `protocol` (TCP/UDP/etc.)
- `length` (packet size)
- `src_port` (source port)

### Why This Matters
- ✅ **More realistic model**: Now learns actual traffic patterns instead of memorizing port mappings
- ⚠️ **Lower accuracy expected**: Accuracy will drop from ~99% to ~60-80% on synthetic data (this is GOOD - it means the model is learning properly)
- 🎯 **Better generalization**: Model should perform better on real-world traffic with different port configurations

---

## 📋 Step-by-Step Migration

### 1. Backup Your Work (Optional but Recommended)
```powershell
# Backup current trained model if you want to keep it
Copy-Item traffic_model.pkl traffic_model_old.pkl
```

### 2. Pull Latest Changes
```powershell
git pull origin main
# or download the updated files
```

### 3. Remove Old Trained Models
```powershell
# Delete old model (trained with dst_port feature)
Remove-Item traffic_model.pkl -ErrorAction SilentlyContinue
Remove-Item deep_model.pt -ErrorAction SilentlyContinue
```

### 4. Retrain Your Model

**Option A: Use existing dataset**
```powershell
.\traffic_env\Scripts\python.exe .\train_model.py --data dataset.csv --model-out traffic_model.pkl
```

**Option B: Capture fresh data and train**
```powershell
# Terminal 1: Start capture (as Administrator)
.\traffic_env\Scripts\python.exe .\capture_features.py --interface 8 --duration 20 --output dataset.csv

# Terminal 2: Generate traffic (wait 2 seconds after starting capture)
.\traffic_env\Scripts\python.exe .\traffic_generator.py --type all --duration 20 --pps 50 --dst 127.0.0.1

# After capture completes, train model
.\traffic_env\Scripts\python.exe .\train_model.py --data dataset.csv --model-out traffic_model.pkl
```

### 5. Verify New Model Works
```powershell
# Test batch prediction
.\traffic_env\Scripts\python.exe .\batch_predict.py --model traffic_model.pkl --data dataset.csv
```

Expected output:
- Accuracy: 60-85% (lower than before - this is expected and GOOD!)
- Classification report showing reasonable precision/recall

---

## 🔄 If You Have Custom Scripts

### Update Feature Lists
If you wrote custom prediction scripts, update them:

**Old (DON'T USE):**
```python
X = df[["protocol", "length", "src_port", "dst_port"]]  # ❌ Includes dst_port
```

**New (CORRECT):**
```python
X = df[["protocol", "length", "src_port"]]  # ✅ No dst_port
```

### Update API Calls
If using `serve_api.py`:

**Old payload:**
```json
{
  "protocol": "TCP",
  "length": 1024,
  "src_port": 50123,
  "dst_port": 6666  // ❌ No longer accepted
}
```

**New payload:**
```json
{
  "protocol": "TCP",
  "length": 1024,
  "src_port": 50123
}
```

---

## 🆕 New Features You Can Use

### 1. Traffic Shaping Safety Features
```powershell
# Preview firewall rules without creating them
.\traffic_env\Scripts\python.exe .\predict_and_shape.py --interface 1 --duration 10 --shape --dry-run

# With automatic cleanup on exit
.\traffic_env\Scripts\python.exe .\predict_and_shape.py --interface 1 --duration 10 --shape

# Keep rules after exit
.\traffic_env\Scripts\python.exe .\predict_and_shape.py --interface 1 --duration 10 --shape --no-cleanup
```

### 2. Flexible Labeling in extract_features.py
```powershell
# Use synthetic labels (VoIP/FTP/HTTP) - default
.\traffic_env\Scripts\python.exe .\packet_capture\extract_features.py --pcap captured_traffic.pcapng --synthetic-labels

# Use real-world labels (Web/DNS/SSH/etc.)
.\traffic_env\Scripts\python.exe .\packet_capture\extract_features.py --pcap captured_traffic.pcapng --real-labels
```

---

## 🐛 Troubleshooting

### Issue: "Model missing required columns"
**Cause**: Trying to use old model with new code (or vice versa)

**Solution**: Retrain the model with the new feature set (see Step 4 above)

### Issue: "Accuracy dropped significantly"
**Expected Behavior**: Accuracy should be 60-85% (not 95%+)

**Explanation**: The old high accuracy was due to label leakage. The new model learns actual traffic patterns, which is harder but more realistic.

**If accuracy < 50%**: 
- Check that you have enough training data (>1000 samples)
- Ensure labels are balanced (similar counts of VoIP/FTP/HTTP)
- Consider running longer capture to get more varied traffic

### Issue: PyTorch import errors
**Solution**: Install PyTorch if you want to use deep learning features:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📊 Understanding New Performance Metrics

### Old Model (with label leakage):
```
Accuracy: 98%  ❌ Too good to be true!
- Memorizing port 5555 → VoIP
- Not learning traffic characteristics
```

### New Model (fixed):
```
Accuracy: 70%  ✅ Realistic!
- Learning from packet size patterns
- Learning from protocol types
- Learning from source port distributions
```

**Better evaluation**: Test on DIFFERENT ports (not 5555/6666/7777) to see true generalization.

---

## 🎯 Next Steps After Migration

1. **Collect more diverse data**: Vary packet sizes, timing, protocols
2. **Add flow-level features** (future enhancement):
   - Packets per second
   - Inter-arrival time
   - Byte distribution
3. **Test on real traffic**: Capture actual VoIP/FTP/HTTP traffic (not synthetic)
4. **Document your findings**: Compare old vs. new model performance

---

## 💬 Need Help?

- Check the updated README.md for usage examples
- Review CHANGELOG.md for all changes
- Open an issue on GitHub if you encounter problems

---

## 🎓 Educational Note

This migration demonstrates an important ML principle: **High accuracy doesn't always mean a good model**. 

The original model with 98% accuracy was actually WORSE than the new 70% model because:
- It cheated by using the label as a feature
- It couldn't generalize to new scenarios
- It wouldn't work on real networks

The new model is honest about what it can and cannot do, making it more valuable for learning and potentially adaptable to real use cases.
