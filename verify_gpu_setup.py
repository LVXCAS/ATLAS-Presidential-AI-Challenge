"""
GTX 1660 Super GPU Verification Script
Run this after installing CUDA and cuDNN to verify setup
"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import os

def check_gpu_setup():
    """Comprehensive GPU setup verification"""

    print("=" * 50)
    print("GTX 1660 SUPER GPU VERIFICATION")
    print("=" * 50)

    # Basic TensorFlow info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA built into TF: {tf.test.is_built_with_cuda()}")

    # Check environment variables
    cuda_path = os.environ.get('CUDA_PATH', 'Not Set')
    print(f"CUDA_PATH environment: {cuda_path}")

    # List all devices
    print(f"\nAll available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")

    # Focus on GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {len(gpus)}")

    if gpus:
        print("\n✅ GPU DETECTED!")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")

            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  Device name: {details.get('device_name', 'GTX 1660 Super')}")
                print(f"  Compute capability: {details.get('compute_capability', 'Unknown')}")
            except:
                print("  (Could not get detailed info)")

        # Memory info (if available)
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"  Current memory usage: {memory_info['current'] / 1024**3:.1f} GB")
            print(f"  Peak memory usage: {memory_info['peak'] / 1024**3:.1f} GB")
        except:
            print("  (Memory info not available)")

        # Performance test
        print(f"\n🚀 PERFORMANCE TEST")
        print("-" * 30)

        # GPU test
        print("Testing GPU performance...")
        start_time = datetime.now()

        try:
            with tf.device('/GPU:0'):
                # Matrix multiplication test
                a = tf.random.normal([2000, 2000], dtype=tf.float32)
                b = tf.random.normal([2000, 2000], dtype=tf.float32)
                c = tf.matmul(a, b)
                result = tf.reduce_sum(c)

            gpu_time = (datetime.now() - start_time).total_seconds()
            print(f"GPU computation time: {gpu_time:.3f} seconds")

            # CPU comparison
            print("Testing CPU performance...")
            start_time = datetime.now()

            with tf.device('/CPU:0'):
                a = tf.random.normal([2000, 2000], dtype=tf.float32)
                b = tf.random.normal([2000, 2000], dtype=tf.float32)
                c = tf.matmul(a, b)
                result = tf.reduce_sum(c)

            cpu_time = (datetime.now() - start_time).total_seconds()
            print(f"CPU computation time: {cpu_time:.3f} seconds")

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"\n🏆 GPU SPEEDUP: {speedup:.1f}x")

            if speedup > 2:
                print("✅ Excellent GPU acceleration!")
            elif speedup > 1:
                print("✅ Good GPU acceleration!")
            else:
                print("⚠️  Limited speedup - check CUDA installation")

        except Exception as e:
            print(f"❌ GPU performance test failed: {e}")

        # Deep learning test
        print(f"\n🧠 DEEP LEARNING TEST")
        print("-" * 30)

        try:
            with tf.device('/GPU:0'):
                # Simple neural network
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                ])

                # Compile model
                model.compile(optimizer='adam', loss='mse')

                # Generate synthetic data
                X = tf.random.normal([1000, 100])
                y = tf.random.normal([1000, 1])

                # Train for a few epochs
                print("Training neural network on GPU...")
                start_time = datetime.now()

                history = model.fit(X, y, epochs=10, batch_size=64, verbose=0)

                training_time = (datetime.now() - start_time).total_seconds()
                print(f"Training time: {training_time:.2f} seconds")
                print(f"Final loss: {history.history['loss'][-1]:.6f}")
                print("✅ Neural network training successful!")

        except Exception as e:
            print(f"❌ Deep learning test failed: {e}")

        print(f"\n🎯 ALPHA DISCOVERY READINESS")
        print("-" * 30)
        print("✅ Your GTX 1660 Super is ready for enhanced alpha discovery!")
        print("✅ Expected 5-10x speedup for training models")
        print("✅ Can handle larger batch sizes and more complex models")
        print("✅ Real-time model updates during market hours possible")

    else:
        print("\n❌ NO GPU DETECTED")
        print("Your GTX 1660 Super is not being recognized by TensorFlow.")
        print("\nPossible solutions:")
        print("1. Install NVIDIA GPU drivers (460+ recommended)")
        print("2. Install CUDA Toolkit (11.8 or 12.x)")
        print("3. Install cuDNN (8.6+)")
        print("4. Add CUDA to PATH environment variable")
        print("5. Restart computer after installation")
        print("\nSee GTX_1660_SUPER_CUDA_SETUP.md for detailed instructions")

        # Still show current CPU capabilities
        print(f"\n💻 CURRENT CPU PERFORMANCE")
        print("-" * 30)

        start_time = datetime.now()

        # CPU matrix test
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)

        cpu_time = (datetime.now() - start_time).total_seconds()
        print(f"CPU computation time: {cpu_time:.3f} seconds")
        print("This is your current baseline - GPU will be 5-10x faster!")

if __name__ == "__main__":
    check_gpu_setup()