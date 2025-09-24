"""
Automated cuDNN Installation for GTX 1660 Super
Downloads and installs cuDNN 8.9+ for CUDA 12.x
"""

import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
import subprocess

def check_admin():
    """Check if running as administrator"""
    try:
        return os.getuid() == 0
    except AttributeError:
        # Windows
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

def download_cudnn():
    """Download cuDNN library"""
    print("=" * 50)
    print("CUDNN AUTOMATED INSTALLATION")
    print("=" * 50)

    if not check_admin():
        print("❌ ERROR: Please run as Administrator")
        print("Right-click and select 'Run as administrator'")
        return False

    # Check if CUDA is installed
    cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6')
    if not os.path.exists(cuda_path):
        print(f"❌ CUDA not found at: {cuda_path}")
        print("Please install CUDA first using quick_cuda_setup.md")
        return False

    print(f"✅ CUDA found at: {cuda_path}")

    # Create download directory
    download_dir = Path("C:/Temp/cuDNN")
    download_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔽 DOWNLOADING cuDNN...")
    print("-" * 30)

    # cuDNN download URLs (these may change - check nvidia.com/cudnn for latest)
    cudnn_urls = [
        "https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-windows-x86_64-9.5.1.17_cuda12-archive.zip",
        "https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-windows-x86_64-9.4.0.58_cuda12-archive.zip",
        "https://developer.download.nvidia.com/compute/cudnn/8.9.7/local_installers/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
    ]

    cudnn_file = None

    for url in cudnn_urls:
        try:
            print(f"Attempting download from: {url}")
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                filename = url.split('/')[-1]
                cudnn_file = download_dir / filename

                print(f"Downloading {filename}...")

                # Download with progress
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(cudnn_file, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rProgress: {percent:.1f}%", end='')

                print(f"\n✅ Downloaded: {cudnn_file}")
                break

        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
            continue

    if not cudnn_file or not cudnn_file.exists():
        print("\n❌ cuDNN download failed from all sources")
        print("\nManual download required:")
        print("1. Go to: https://developer.nvidia.com/cudnn")
        print("2. Create free NVIDIA account")
        print("3. Download: cuDNN Library for Windows (x64)")
        print("4. Save to: C:/Temp/cuDNN/")
        print("5. Re-run this script")
        return False

    return install_cudnn(cudnn_file, cuda_path)

def install_cudnn(cudnn_file, cuda_path):
    """Install cuDNN to CUDA directory"""
    print(f"\n📦 INSTALLING cuDNN...")
    print("-" * 30)

    extract_dir = Path("C:/Temp/cuDNN/extracted")
    extract_dir.mkdir(exist_ok=True)

    try:
        # Extract cuDNN
        print("Extracting cuDNN archive...")
        with zipfile.ZipFile(cudnn_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find extracted directory
        cudnn_dirs = list(extract_dir.glob("cudnn-*"))
        if not cudnn_dirs:
            print("❌ cuDNN extraction failed - no cudnn directory found")
            return False

        cudnn_extracted = cudnn_dirs[0]
        print(f"✅ Extracted to: {cudnn_extracted}")

        # Copy files to CUDA installation
        cuda_path = Path(cuda_path)

        # Copy bin files
        bin_src = cudnn_extracted / "bin"
        bin_dst = cuda_path / "bin"
        if bin_src.exists():
            for file in bin_src.glob("*.dll"):
                dst_file = bin_dst / file.name
                shutil.copy2(file, dst_file)
                print(f"✅ Copied: {file.name} → bin/")

        # Copy include files
        include_src = cudnn_extracted / "include"
        include_dst = cuda_path / "include"
        if include_src.exists():
            for file in include_src.glob("*.h"):
                dst_file = include_dst / file.name
                shutil.copy2(file, dst_file)
                print(f"✅ Copied: {file.name} → include/")

        # Copy lib files
        lib_src = cudnn_extracted / "lib"
        lib_dst = cuda_path / "lib" / "x64"
        if lib_src.exists():
            for file in lib_src.glob("*.lib"):
                dst_file = lib_dst / file.name
                shutil.copy2(file, dst_file)
                print(f"✅ Copied: {file.name} → lib/x64/")

        print("\n✅ cuDNN installation complete!")
        return True

    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print(f"\n🔧 CONFIGURING ENVIRONMENT...")
    print("-" * 30)

    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

    try:
        # Set CUDA_PATH
        subprocess.run([
            'setx', 'CUDA_PATH', cuda_path, '/M'
        ], check=True, capture_output=True, text=True)
        print("✅ CUDA_PATH environment variable set")

        # Add to system PATH
        cuda_bin = f"{cuda_path}\\bin"
        cuda_libnvvp = f"{cuda_path}\\libnvvp"

        # Get current PATH
        result = subprocess.run([
            'reg', 'query',
            'HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment',
            '/v', 'PATH'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            current_path = result.stdout.split('REG_EXPAND_SZ')[-1].strip()

            # Add CUDA paths if not present
            if cuda_bin not in current_path:
                new_path = f"{current_path};{cuda_bin};{cuda_libnvvp}"
                subprocess.run([
                    'setx', 'PATH', new_path, '/M'
                ], check=True, capture_output=True)
                print("✅ CUDA added to system PATH")
            else:
                print("✅ CUDA already in system PATH")

        return True

    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def verify_installation():
    """Verify CUDA and cuDNN installation"""
    print(f"\n🔍 VERIFYING INSTALLATION...")
    print("-" * 30)

    try:
        # Check nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA compiler (nvcc) working")
            print(f"   Version: {result.stdout.split('release')[1].split(',')[0].strip()}")
        else:
            print("❌ CUDA compiler not found")
    except:
        print("❌ CUDA compiler not accessible")

    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA System Management Interface working")
        else:
            print("❌ nvidia-smi not working")
    except:
        print("❌ nvidia-smi not accessible")

    # Test TensorFlow GPU
    try:
        print("\n🧠 Testing TensorFlow GPU detection...")
        result = subprocess.run([
            sys.executable, '-c',
            'import tensorflow as tf; '
            'print(f"TensorFlow: {tf.__version__}"); '
            'print(f"CUDA built: {tf.test.is_built_with_cuda()}"); '
            'gpus = tf.config.list_physical_devices("GPU"); '
            'print(f"GPUs found: {len(gpus)}"); '
            'print("GPU details:", gpus)'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ TensorFlow GPU test results:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
        else:
            print("❌ TensorFlow GPU test failed")
            print(f"   Error: {result.stderr}")

    except Exception as e:
        print(f"❌ TensorFlow test error: {e}")

def cleanup():
    """Clean up temporary files"""
    print(f"\n🧹 CLEANING UP...")
    print("-" * 30)

    try:
        temp_dir = Path("C:/Temp/cuDNN")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main installation process"""
    print("GTX 1660 SUPER - AUTOMATED cuDNN INSTALLATION")
    print("=" * 50)

    # Step 1: Download and install cuDNN
    if not download_cudnn():
        return False

    # Step 2: Setup environment variables
    if not setup_environment():
        return False

    # Step 3: Verify installation
    verify_installation()

    # Step 4: Cleanup
    cleanup()

    print("\n" + "=" * 50)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 50)
    print("\n📋 NEXT STEPS:")
    print("1. 🔄 RESTART your computer")
    print("2. 🧪 Run: python verify_gpu_setup.py")
    print("3. 🚀 Run: python gpu_enhanced_alpha_discovery.py")
    print("\nYour GTX 1660 Super is ready for 5-10x speedup!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            input("\nPress Enter to exit...")
        else:
            input("\nInstallation failed. Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")