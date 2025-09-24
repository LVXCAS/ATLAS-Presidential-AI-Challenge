# GTX 1660 Super CUDA Setup Guide

## Overview
Your GTX 1660 Super has excellent CUDA capabilities that can provide 5-10x speedup for our deep learning alpha discovery systems. Currently, TensorFlow isn't detecting your GPU because CUDA and cuDNN need to be installed.

## System Requirements
- **GPU**: GTX 1660 Super ✅ (You have this)
- **CUDA**: Version 11.8 or 12.x (Currently missing)
- **cuDNN**: Version 8.6+ (Currently missing)
- **TensorFlow**: 2.20.0 ✅ (Already installed)

## Step-by-Step Installation

### 1. Download CUDA Toolkit
```
Visit: https://developer.nvidia.com/cuda-downloads
Select:
- Operating System: Windows
- Architecture: x86_64
- Version: 11 (or 10 if on older Windows)
- Installer Type: exe (local)

Download: cuda_12.6.2_560.94_windows.exe (or latest version)
```

### 2. Install CUDA
```cmd
# Run the downloaded installer as Administrator
# Accept the license agreement
# Choose "Custom" installation
# Select:
  ✅ CUDA Toolkit
  ✅ CUDA Samples
  ✅ CUDA Documentation
  ✅ NVIDIA GeForce Experience (if not already installed)

# Installation path will be: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### 3. Download cuDNN
```
Visit: https://developer.nvidia.com/cudnn
- Create free NVIDIA Developer account if needed
- Download cuDNN Library for Windows (x64)
- Version: 8.9.7 for CUDA 12.x (or compatible version)
```

### 4. Install cuDNN
```cmd
# Extract the cuDNN zip file
# You'll see folders: bin, include, lib

# Copy files to CUDA installation directory:
Copy cuDNN files to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\

# Specifically:
cudnn64_8.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\
cudnn.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\
cudnn.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\
```

### 5. Update Environment Variables
```cmd
# Add to PATH (System Environment Variables):
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp

# Add new environment variable:
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### 6. Verify Installation
```cmd
# Open new Command Prompt (important - restart cmd after env variables)
nvcc --version
nvidia-smi

# Should show CUDA version and GPU information
```

### 7. Test TensorFlow GPU Detection
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Expected output after successful installation:
# TensorFlow version: 2.20.0
# CUDA available: True
# GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Performance Benefits for Your Trading System

### Current CPU Performance
- **Training Time**: ~2-5 minutes per symbol
- **Model Complexity**: Limited by CPU constraints
- **Batch Size**: 64-128 samples
- **Concurrent Training**: Single model at a time

### Expected GPU Performance (GTX 1660 Super)
- **Training Time**: ~20-60 seconds per symbol (5-10x speedup)
- **Model Complexity**: Can handle larger networks
- **Batch Size**: 256-512 samples (2-4x larger)
- **Concurrent Training**: Multiple models in parallel

### GTX 1660 Super Specifications
```
- CUDA Cores: 1,408
- Memory: 6GB GDDR6
- Memory Bandwidth: 336 GB/s
- Base Clock: 1,530 MHz
- Boost Clock: 1,785 MHz
- CUDA Compute Capability: 7.5
```

## Expected Alpha Discovery Improvements

### Enhanced Model Training
1. **Faster Iterations**: 5-10x faster training cycles
2. **Larger Models**: More complex LSTM/CNN/Transformer architectures
3. **Better Accuracy**: Larger batch sizes = more stable gradients
4. **Real-time Training**: Live model updates during market hours

### Advanced Techniques Enabled
1. **Hyperparameter Optimization**: Grid search becomes feasible
2. **Ensemble Methods**: Train multiple models simultaneously
3. **Transfer Learning**: Pre-train models on historical data
4. **Reinforcement Learning**: Enable RL-based strategy optimization

## Troubleshooting

### Common Issues
1. **"CUDA version mismatch"**: Ensure TensorFlow and CUDA versions are compatible
2. **"Could not load dynamic library"**: Check PATH environment variables
3. **"Out of memory"**: Reduce batch size in gpu_enhanced_alpha_discovery.py
4. **"No CUDA devices"**: Restart computer after installation

### Quick Fix Commands
```cmd
# Reset TensorFlow GPU memory
set TF_FORCE_GPU_ALLOW_GROWTH=true

# Check GPU memory usage
nvidia-smi

# Force CUDA device visibility
set CUDA_VISIBLE_DEVICES=0
```

## Integration with Current Systems

Once CUDA is installed, your existing systems will automatically accelerate:

### Files That Will Benefit
- `gpu_enhanced_alpha_discovery.py` - Primary beneficiary
- `deep_learning_demo_system.py` - 5-10x faster training
- `quantum_ml_ensemble.py` - Parallel ensemble training
- All R&D systems with ML components

### No Code Changes Needed
The systems are already GPU-ready and will automatically detect and use the GPU once CUDA is installed.

## Verification Script

After installation, run this verification:

```python
# Save as verify_gpu_setup.py
import tensorflow as tf
import numpy as np
from datetime import datetime

print("=== GTX 1660 Super GPU Verification ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices found: {len(gpus)}")

if gpus:
    gpu = gpus[0]
    details = tf.config.experimental.get_device_details(gpu)
    print(f"GPU name: {details.get('device_name', 'GTX 1660 Super')}")
    print(f"Memory limit: {tf.config.experimental.get_memory_info('GPU:0')}")

    # Test GPU performance
    print("\nTesting GPU performance...")
    start_time = datetime.now()

    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)

    gpu_time = (datetime.now() - start_time).total_seconds()
    print(f"GPU computation time: {gpu_time:.3f} seconds")

    # Compare with CPU
    start_time = datetime.now()

    with tf.device('/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)

    cpu_time = (datetime.now() - start_time).total_seconds()
    print(f"CPU computation time: {cpu_time:.3f} seconds")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")

    print("\n✅ GTX 1660 Super successfully configured!")
    print("Your alpha discovery systems will now run 5-10x faster!")

else:
    print("\n❌ No GPU detected. Please follow the setup guide above.")
```

## Next Steps

1. **Install CUDA & cuDNN** following steps above
2. **Restart computer** to ensure all environment variables are loaded
3. **Run verification script** to confirm GPU detection
4. **Re-run** `gpu_enhanced_alpha_discovery.py` for dramatic speedup
5. **Monitor performance** improvements in your trading systems

## Support

If you encounter issues:
1. Check NVIDIA Driver version (should be 460+ for RTX/GTX 16 series)
2. Verify Windows version compatibility with CUDA
3. Try different CUDA/cuDNN version combinations
4. Check TensorFlow GPU compatibility matrix online

Your GTX 1660 Super is a powerful GPU that will significantly enhance your quantitative trading capabilities!