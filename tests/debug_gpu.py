
import sys
import os
import traceback

def debug_gpu():
    print("--- GPU DIAGNOSTIC ---")
    
    # 1. Check Python Environment
    print(f"Python: {sys.version}")
    
    # 2. Try Import CuPy
    print("\nAttempting to import cupy...")
    try:
        import cupy
        print(f"SUCCESS: CuPy {cupy.__version__} imported.")
        print(f"CUDA Path: {cupy.cuda.get_cuda_path()}")
        
        # Try a small allocation
        import cupy as cp
        a = cp.array([1, 2, 3])
        print(f"Allocation Test: {a}")
        
    except ImportError as e:
        print(f"FAILED: ImportError.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"FAILED: Exception.")
        traceback.print_exc()

if __name__ == "__main__":
    debug_gpu()
