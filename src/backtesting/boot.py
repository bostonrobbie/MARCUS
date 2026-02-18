import sys
import os
import glob
import warnings

# --- 0. Windows Unicode Fix (Issue #8) ---
# Windows console can't display emojis/unicode by default
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# --- 1. Path Configuration ---
# Ensure 'src' is in python path regardless of where script is run
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/backtesting
src_root = os.path.dirname(current_dir) # src
project_root = os.path.dirname(src_root) # StrategyPipeline

if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Inject Project Root to PATH for DLL Hacks (nvrtc shim)
os.environ['PATH'] = project_root + ";" + os.environ['PATH']
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(project_root)
    except Exception:
        pass

# --- 2. Auto-CUDA Fixer ---
# Cupy on Windows often fails to find nvrtc DLLs even if installed.
# We hunt for them and add them to PATH.
def _inject_cuda_path():
    try:
        # Standard Location
        base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if not os.path.exists(base_path):
            return

        # Find all versions (v11.x, v12.x)
        versions = glob.glob(os.path.join(base_path, "v*"))
        
        candidates = []
        for v_dir in versions:
            bin_dir = os.path.join(v_dir, "bin")
            if os.path.exists(bin_dir):
                candidates.append(bin_dir)
                
        # Also check for independent NVRTC installations or Conda envs? 
        # For now, just NVIDIA Toolkit.
        
        if candidates:
            # Sort by version descending (newest first)
            candidates.sort(reverse=True)
            
            # Add to PATH
            current_path = os.environ.get('PATH', '')
            for bin_dir in candidates:
                # Add bin
                if bin_dir not in current_path:
                    # print(f"[BOOT] Injecting CUDA to PATH: {bin_dir}")
                    os.environ['PATH'] = bin_dir + ";" + os.environ['PATH']
                
                # Add bin/x64 (Common in newer CUDA for DLLs)
                bin_x64 = os.path.join(bin_dir, "x64")
                if os.path.exists(bin_x64) and bin_x64 not in current_path:
                    os.environ['PATH'] = bin_x64 + ";" + os.environ['PATH']
            
    except Exception as e:
        print(f"[BOOT] Warning: CUDA injection failed: {e}")

_inject_cuda_path()

# --- 3. Noise Cancellation ---
# Suppress useless warnings that scare users
warnings.filterwarnings('ignore', category=UserWarning, module='cupy')
warnings.filterwarnings('ignore', message='.*CUDA path could not be detected.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Global success flag
BOOT_SUCCESS = True
