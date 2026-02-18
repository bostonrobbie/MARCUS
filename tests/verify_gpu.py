
import sys
import os
import glob
import time
import numpy as np

# --- AGGRESSIVE DLL REGISTRATION ---
import site
site_packages = site.getsitepackages()[1]
nvidia_path = os.path.join(site_packages, 'nvidia')

print(f"Scanning {nvidia_path} for DLLs...")

dll_dirs = []
for root, dirs, files in os.walk(nvidia_path):
    if 'bin' in dirs:
        bin_path = os.path.join(root, 'bin')
        dll_dirs.append(bin_path)
    if 'lib' in dirs:
        lib_path = os.path.join(root, 'lib')
        dll_dirs.append(lib_path)

# Register ALL found directories
for d in dll_dirs:
    # 1. Add to PATH (Legacy)
    os.environ['PATH'] += os.pathsep + d
    
    # 2. Add via os.add_dll_directory (Modern)
    try:
        os.add_dll_directory(d)
        print(f"Registered DLL Dir: {d}")
    except Exception as e:
        print(f"Failed to register {d}: {e}")

# --- IMPORT CUPY ---
print("Importing CuPy...")
try:
    import cupy as cp
    HAS_CUPY = True
    print(f"CuPy Imported! Version: {cp.__version__}")
except ImportError as e:
    print(f"CuPy Import Failed: {e}")
    HAS_CUPY = False
except Exception as e:
    print(f"CuPy Error: {e}")
    HAS_CUPY = False
    
def test_gpu_indicators():
    if not HAS_CUPY:
        print("Skipping GPU Test.")
        return

    # Generate massive random data
    N = 100_000
    print(f"Generating {N} data points on GPU...")
    try:
        prices_gpu = cp.random.random(N).astype(cp.float64) * 100
        
        print("Testing GPU Kernel (SMA)...")
        # Simple Kernel
        kernel = cp.ones(50) / 50
        sma = cp.convolve(prices_gpu, kernel, mode='valid')
        
        # Force Sync
        cp.cuda.Stream.null.synchronize()
        print("GPU Computation Success!")
        print(f"Result Shape: {sma.shape}")
        
    except Exception as e:
        print(f"GPU Computation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_indicators()
