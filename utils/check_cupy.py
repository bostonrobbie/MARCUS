try:
    import cupy
    print(f"CuPy Available: {cupy.__version__}")
    print("GPU Device:", cupy.cuda.Device(0).compute_capability)
except ImportError:
    print("CuPy NOT installed.")
except Exception as e:
    print(f"Error checking CuPy: {e}")
