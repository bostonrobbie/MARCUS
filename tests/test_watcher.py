"""
NHHF Automatic Test Watcher
===========================
Monitors for code changes and runs tests automatically.
Sends notifications when tests fail.

Usage:
    python test_watcher.py

Press Ctrl+C to stop watching.
"""

import subprocess
import sys
import time
import os
import json
from datetime import datetime
from pathlib import Path
import hashlib

# Configuration
WATCH_PATHS = [
    "../StrategyPipeline/src",
    "../Fund_Manager",
    "."  # tests folder itself
]
CHECK_INTERVAL = 30  # seconds between checks
NOTIFICATION_FILE = "../Fund_Manager/reports/TEST_NOTIFICATIONS.json"

# File extensions to watch
WATCH_EXTENSIONS = {'.py'}


def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file contents."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


def get_all_file_hashes(base_path: Path) -> dict:
    """Get hashes of all Python files in directory tree."""
    hashes = {}
    for root, dirs, files in os.walk(base_path):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'node_modules', 'venv', 'env', 'gpu_env'}]

        for file in files:
            if Path(file).suffix in WATCH_EXTENSIONS:
                filepath = Path(root) / file
                hashes[str(filepath)] = get_file_hash(filepath)
    return hashes


def run_tests() -> tuple[bool, str]:
    """Run pytest and return success status and output."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-v', '--tb=short', '-q'],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"


def send_notification(success: bool, message: str, changed_files: list):
    """
    Send notification about test results.
    Writes to notification file for dashboard to pick up.
    Also shows Windows toast notification if available.
    """
    notification = {
        "timestamp": datetime.now().isoformat(),
        "status": "PASS" if success else "FAIL",
        "message": message[:500],  # Truncate long messages
        "changed_files": changed_files[:10],  # Limit to 10 files
        "test_output": message[-2000:] if not success else ""  # Last 2000 chars on failure
    }

    # Write to notification file
    notification_path = Path(__file__).parent / NOTIFICATION_FILE
    notification_path.parent.mkdir(parents=True, exist_ok=True)

    notifications = []
    if notification_path.exists():
        try:
            with open(notification_path, 'r') as f:
                notifications = json.load(f)
        except:
            notifications = []

    notifications.insert(0, notification)  # Add to front
    notifications = notifications[:50]  # Keep last 50

    with open(notification_path, 'w') as f:
        json.dump(notifications, f, indent=2)

    # Console notification
    icon = "[PASS]" if success else "[FAIL]"
    color = "\033[92m" if success else "\033[91m"  # Green or Red
    reset = "\033[0m"

    print(f"\n{color}{'='*60}")
    print(f" {icon} TEST RESULTS - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}{reset}")

    if changed_files:
        print(f"Changed files: {', '.join(changed_files[:5])}")

    if not success:
        print(f"\n{color}TESTS FAILED!{reset}")
        # Show last few lines of output
        lines = message.strip().split('\n')
        for line in lines[-15:]:
            print(f"  {line}")
    else:
        print(f"{color}All tests passed!{reset}")

    print()

    # Windows toast notification (if available)
    try:
        if sys.platform == 'win32':
            import ctypes
            if not success:
                ctypes.windll.user32.MessageBeep(0x00000030)  # Warning beep
    except:
        pass


def watch_and_test():
    """Main watch loop."""
    base_paths = [Path(__file__).parent / p for p in WATCH_PATHS]

    print("\n" + "="*60)
    print(" NHHF Test Watcher - Automatic Test Runner")
    print("="*60)
    print(f" Watching: {', '.join(WATCH_PATHS)}")
    print(f" Check interval: {CHECK_INTERVAL} seconds")
    print(" Press Ctrl+C to stop")
    print("="*60 + "\n")

    # Get initial file hashes
    file_hashes = {}
    for base_path in base_paths:
        if base_path.exists():
            file_hashes.update(get_all_file_hashes(base_path))

    print(f"[INIT] Tracking {len(file_hashes)} Python files")

    # Run tests once on startup
    print("[INIT] Running initial test suite...")
    success, output = run_tests()
    send_notification(success, output, [])

    last_check = time.time()

    while True:
        try:
            time.sleep(1)

            # Check for changes every CHECK_INTERVAL seconds
            if time.time() - last_check < CHECK_INTERVAL:
                continue

            last_check = time.time()

            # Get current file hashes
            current_hashes = {}
            for base_path in base_paths:
                if base_path.exists():
                    current_hashes.update(get_all_file_hashes(base_path))

            # Find changed files
            changed_files = []
            for filepath, hash_val in current_hashes.items():
                if filepath not in file_hashes or file_hashes[filepath] != hash_val:
                    changed_files.append(Path(filepath).name)

            # Check for deleted files
            for filepath in file_hashes:
                if filepath not in current_hashes:
                    changed_files.append(f"{Path(filepath).name} (deleted)")

            if changed_files:
                print(f"\n[CHANGE] Detected changes in: {', '.join(changed_files[:5])}")
                print("[TEST] Running test suite...")

                success, output = run_tests()
                send_notification(success, output, changed_files)

                # Update hashes
                file_hashes = current_hashes

        except KeyboardInterrupt:
            print("\n[STOP] Test watcher stopped by user")
            break


if __name__ == '__main__':
    watch_and_test()
