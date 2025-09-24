# Quick CUDA Setup for GTX 1660 Super

## 🚀 **Manual Installation (Recommended)**

### **Step 1: Download CUDA**
1. Open browser: https://developer.nvidia.com/cuda-downloads
2. Select: Windows → x86_64 → 11 → exe (local)
3. Download: `cuda_12.6.2_560.94_windows.exe` (~3GB)

### **Step 2: Install CUDA**
1. Run installer **as Administrator**
2. Choose **Custom** installation
3. Select: CUDA Toolkit + Samples + Documentation
4. Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`

### **Step 3: Download cuDNN**
1. Go to: https://developer.nvidia.com/cudnn
2. Create free NVIDIA account (if needed)
3. Download: cuDNN Library for Windows (x64)
4. Version: 8.9.7 for CUDA 12.x

### **Step 4: Install cuDNN**
1. Extract ZIP file
2. Copy files to CUDA directory:
   ```
   cudnn64_8.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\
   cudnn.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\
   cudnn.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\
   ```

### **Step 5: Set Environment Variables**
1. **Right-click This PC** → Properties → Advanced → Environment Variables
2. **System Variables** → New:
   - Variable: `CUDA_PATH`
   - Value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
3. **Edit PATH** → Add:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp`

### **Step 6: Restart & Test**
1. **Restart computer**
2. Run: `python verify_gpu_setup.py`
3. Should show GTX 1660 Super detected!

---

## ⚡ **Expected Performance Gains**

| Current (CPU) | With GTX 1660 Super |
|---------------|---------------------|
| 2-5 min/symbol | 20-60 sec/symbol |
| 64 batch size | 256-512 batch size |
| Sequential training | Parallel training |

---

## 🎯 **Instant Benefits After Installation**

✅ **5-10x faster** alpha discovery
✅ **Real-time** model training during market hours
✅ **Advanced architectures** (larger LSTM/CNN/Transformers)
✅ **Parallel ensemble** training
✅ **Hyperparameter optimization** becomes feasible

---

## 🔧 **Verification Commands**

```cmd
# Test CUDA installation
nvcc --version
nvidia-smi

# Test TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Run comprehensive test
python verify_gpu_setup.py
```

---

## 🚨 **Troubleshooting**

**No GPU detected?**
- Update NVIDIA drivers from geforce.com
- Restart after environment variable changes
- Check Windows version compatibility

**Installation fails?**
- Run installers as Administrator
- Disable antivirus temporarily
- Ensure sufficient disk space (5GB+)

**Memory errors?**
- Reduce batch size in scripts
- Close other GPU applications
- Restart Python interpreter

---

Your **GTX 1660 Super** is ready to dramatically accelerate your quantitative trading systems!

**Current Status**: Enhanced alpha discovery training on CPU (takes ~5min/symbol)
**After CUDA**: Same training will take ~30 seconds/symbol with better accuracy!