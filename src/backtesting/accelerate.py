"""
GPU Acceleration Utilities for Backtesting

This module provides conditional GPU support using NVIDIA RAPIDS (cuDF, cuPy).
If CUDA/RAPIDS is not available, it falls back to CPU (pandas/numpy).
"""

import sys

# --- GPU Detection and Backend Selection ---

GPU_AVAILABLE = False
_gpu_info = {}

def _check_gpu():
    """Check if NVIDIA GPU and RAPIDS are available."""
    global GPU_AVAILABLE, _gpu_info
    
    # Check for cuDF
    try:
        import cudf
        _gpu_info['cudf'] = cudf.__version__
    except ImportError:
        _gpu_info['cudf'] = None
    
    # Check for cuPy
    try:
        import cupy
        _gpu_info['cupy'] = cupy.__version__
        
        # Verify GPU is actually accessible
        try:
            _ = cupy.cuda.Device(0).compute_capability
            _gpu_info['cuda_device'] = True
        except Exception as e:
            _gpu_info['cuda_device'] = False
            _gpu_info['error'] = str(e)
    except ImportError:
        _gpu_info['cupy'] = None
        _gpu_info['cuda_device'] = False
    
    GPU_AVAILABLE = _gpu_info.get('cuda_device', False) and _gpu_info.get('cupy') is not None
    return GPU_AVAILABLE

# Run check on import
_check_gpu()

def get_gpu_info():
    """Returns GPU availability and version info."""
    return {
        'gpu_available': GPU_AVAILABLE,
        **_gpu_info
    }

def print_gpu_status():
    """Prints GPU status to console."""
    info = get_gpu_info()
    print("\n--- GPU Status ---")
    if info['gpu_available']:
        print(f"[OK] GPU Acceleration Available")
        print(f"     cuDF: {info.get('cudf', 'N/A')}")
        print(f"     cuPy: {info.get('cupy', 'N/A')}")
    else:
        print("[--] GPU Not Available (Using CPU)")
        if info.get('cudf') is None:
            print("     Missing: cudf (pip install cudf-cu12)")
        if info.get('cupy') is None:
            print("     Missing: cupy (pip install cupy-cuda12x)")
        if 'error' in info:
            print(f"     Error: {info['error']}")
    print("------------------\n")

# --- Data Backend Selection ---

def get_dataframe_library():
    """
    Returns cuDF if available, else pandas.
    Usage: pd = get_dataframe_library()
    """
    if GPU_AVAILABLE:
        try:
            import cudf
            return cudf
        except Exception:
            pass

    import pandas
    return pandas

def get_array_library():
    """
    Returns cuPy if available, else numpy.
    Usage: np = get_array_library()
    """
    if GPU_AVAILABLE:
        try:
            import cupy
            return cupy
        except Exception:
            pass

    import numpy
    return numpy

# --- GPU-Accelerated Monte Carlo ---

def gpu_monte_carlo(returns, n_sims=1000, initial_equity=100000.0):
    """
    Runs Monte Carlo simulation using GPU if available.
    """
    xp = get_array_library()
    
    try:
        # Convert returns to xp array
        ret_arr = xp.array(returns)
        n = len(ret_arr)
        
        # Bootstrap resampling ( vectorized )
        # Note: cupy.random.randint might fail if curand is missing
        indices = xp.random.randint(0, n, size=(n_sims, n))
        sim_returns = ret_arr[indices]
        
        # Cumulative returns across each simulation (axis 1)
        # Prod of (1+r)
        sim_final_equity = initial_equity * (1 + sim_returns).prod(axis=1)
        
        # Convert back to numpy for processing
        if hasattr(sim_final_equity, 'get'):
            final_arr = sim_final_equity.get()
        else:
            final_arr = sim_final_equity
            
    except Exception as e:
        print(f"GPU Monte Carlo failed: {e}. Falling back to CPU...")
        import numpy as np
        ret_arr = np.array(returns)
        n = len(ret_arr)
        indices = np.random.randint(0, n, size=(n_sims, n))
        sim_returns = ret_arr[indices]
        final_arr = initial_equity * (1 + sim_returns).prod(axis=1)

    import numpy as np
    return {
        'p5': float(np.percentile(final_arr, 5)),
        'p50': float(np.percentile(final_arr, 50)),
        'p95': float(np.percentile(final_arr, 95)),
        'mean': float(np.mean(final_arr)),
        'std': float(np.std(final_arr)),
        'final_values': final_arr.tolist()
    }
