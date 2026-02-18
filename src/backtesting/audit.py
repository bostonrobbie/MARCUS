
import os
import json
import sys
from typing import Dict, List, Tuple

class PreFlightCheck:
    """
    Gatekeeper that verifies system integrity before execution.
    Checks: GPU, Config, Data Dirs, Permissions.
    """
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.status = []
        self.errors = []
        self.warnings = []

    def run(self) -> bool:
        """Runs all checks. Returns True if Critical Checks Pass."""
        print("\n🔎 Running Pre-Flight Checks...")
        
        self.check_config()
        self.check_gpu()
        self.check_filesystem()
        
        # Report
        print(f"   Checks Complete: {len(self.status)} Run.")
        if self.errors:
            print(f"   ❌ CRITICAL FAILURES ({len(self.errors)}):")
            for e in self.errors: print(f"      - {e}")
            return False
        elif self.warnings:
            print(f"   ⚠️  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings: print(f"      - {w}")
            print("   ✅ System GO (with warnings).")
            return True
        else:
            print("   ✅ All Systems GO.")
            return True

    def check_config(self):
        if not os.path.exists(self.config_path):
            self.errors.append(f"Missing Config: {self.config_path} not found.")
            return
            
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Schema check
                if 'data' not in config:
                    self.warnings.append("Config missing 'data' section.")
        except json.JSONDecodeError:
            self.errors.append(f"Corrupted Config: {self.config_path} is not valid JSON.")

    def check_gpu(self):
        try:
            import numba
            from numba import cuda
            if cuda.is_available():
                self.status.append("GPU: Available")
            else:
                self.warnings.append("GPU: Not Detected (numba.cuda.is_available() = False). Using CPU.")
        except ImportError:
             self.warnings.append("GPU: drivers/modules missing. Using CPU.")

    def check_filesystem(self):
        # 1. Output Permissions
        dirs = ['logs', 'reports', 'outputs', 'cache']
        for d in dirs:
            try:
                os.makedirs(d, exist_ok=True)
                test_file = os.path.join(d, '.test')
                with open(test_file, 'w') as f: f.write('test')
                os.remove(test_file)
            except OSError as e:
                self.errors.append(f"Permission Denied: Cannot write to {d}. {e}")

        # 2. Data Availability
        # We don't fail here because DataHandler can download, but we warn
        pass

if __name__ == "__main__":
    check = PreFlightCheck()
    if not check.run():
        sys.exit(1)
