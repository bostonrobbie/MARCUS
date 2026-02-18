
import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import numba
    print(f"Numba Version: {numba.__version__}")
    
    from numba import cuda
    print(f"CUDA Available: {cuda.is_available()}")
    
    if cuda.is_available():
        print("Detecting GPUs...")
        for gpu in cuda.gpus:
            print(f"GPU: {gpu.name}")
    else:
        print("CUDA is NOT available (numba.cuda.is_available() returned False).")
        print("Checking CUDA system info...")
        try:
            cuda.detect()
        except Exception as e:
            print(f"Error checking CUDA detect: {e}")

except ImportError as e:
    print(f"CRITICAL: Numba Import Failed!")
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred:")
    print(e)

print("\nChecking Environment Variables:")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not Set')}")
print(f"NUMBA_CUDA_DRIVER: {os.environ.get('NUMBA_CUDA_DRIVER', 'Not Set')}")
print(f"PATH (Truncated): {os.environ.get('PATH', '')[:500]}...")
