import os
import sys
import pandas as pd
import importlib
from datetime import datetime
from .accelerate import get_gpu_info, get_dataframe_library, get_array_library
from .data import SmartDataHandler

class SystemDiagnostics:
    """
    Comprehensive System Health & QA Checker.
    """
    def __init__(self, data_root):
        self.data_root = data_root
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'platform': sys.platform,
            'python_version': sys.version,
            'checks': []
        }
        
    def run_all(self):
        print("Running System Diagnostics...")
        self.check_python_env()
        self.check_gpu_health()
        self.check_data_integrity()
        self.check_strategy_imports()
        return self.report

    def log(self, category, status, message, details=None):
        entry = {
            'category': category,
            'status': status,
            'message': message,
            'details': details
        }
        self.report['checks'].append(entry)
        icon = "✅" if status == 'PASS' else ("⚠️" if status == 'WARNING' else "❌")
        print(f"{icon} [{category}] {message}")

    def check_python_env(self):
        # Check critical libraries
        reqs = ['numpy', 'pandas', 'matplotlib']
        for lib in reqs:
            try:
                importlib.import_module(lib)
                self.log('Environment', 'PASS', f"Library '{lib}' installed.")
            except ImportError as e:
                self.log('Environment', 'FAIL', f"Library '{lib}' missing.", str(e))
                
        # Check Numba (Optional but good)
        try:
            import numba
            self.log('Environment', 'PASS', f"Numba {numba.__version__} installed (Fast CPU).")
        except ImportError:
            self.log('Environment', 'WARNING', "Numba not found. CPU Backtests will be slow.")

    def check_gpu_health(self):
        info = get_gpu_info()
        if info['gpu_available']:
            self.log('Hardware', 'PASS', f"GPU Detected: {info.get('cupy')}")
            # Try a small cupy operation
            try:
                xp = get_array_library()
                a = xp.array([1, 2, 3])
                b = a * 2
                if int(b[0]) == 2:
                    self.log('Hardware', 'PASS', "GPU Calculation Verified (1+1=2).")
                else:
                    self.log('Hardware', 'FAIL', "GPU Calculation Verification Failed.")
            except Exception as e:
                self.log('Hardware', 'FAIL', "GPU Runtime Error during validation.", str(e))
        else:
            self.log('Hardware', 'WARNING', "GPU Not Available. Falling back to CPU.")
            if info.get('error'):
                self.log('Hardware', 'INFO', f"GPU Error Details: {info['error']}")

    def check_data_integrity(self):
        if not os.path.exists(self.data_root):
             self.log('Data', 'FAIL', f"Data Root '{self.data_root}' not found.")
             return
             
        csv_files = [f for f in os.listdir(self.data_root) if f.endswith('.csv')]
        if not csv_files:
            self.log('Data', 'WARNING', "No CSV files found in data directory.")
            return
            
        self.log('Data', 'PASS', f"Found {len(csv_files)} CSV files.")
        
        # Check Schema of first few
        for f in csv_files[:3]:
            path = os.path.join(self.data_root, f)
            try:
                df = pd.read_csv(path, nrows=5)
                # Check required columns
                req_cols = ['open', 'high', 'low', 'close', 'volume']
                cols = [c.lower() for c in df.columns]
                missing = [rc for rc in req_cols if rc not in cols]
                
                if missing:
                    self.log('Data', 'FAIL', f"File {f} missing columns: {missing}", {'columns': cols})
                else:
                    self.log('Data', 'PASS', f"File {f} schema valid.")
            except Exception as e:
                self.log('Data', 'FAIL', f"File {f} unreadable.", str(e))

    def check_strategy_imports(self):
        # Try to import strategies
        strategies = ['MovingAverageCrossover', 'NqOrb', 'NqOrb15m', 'NqOrbEnhanced']
        # We need to ensure we can find them. Assuming execution from strategies dir or sys path setup.
        pass # Difficult to do generically without strict paths.
