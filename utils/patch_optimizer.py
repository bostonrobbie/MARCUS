import os

file_path = r"C:\Users\User\Documents\AI\backtesting\optimizer.py"

with open(file_path, 'r') as f:
    content = f.read()

# Target string (exact match from previous view)
target = """                 initial_capital: float = 100000.0,
                 n_jobs: int = -1):"""

replacement = """                 initial_capital: float = 100000.0,
                 n_jobs: int = -1,
                 vector_strategy_cls=None):"""

if target in content:
    new_content = content.replace(target, replacement)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully patched optimizer.py")
else:
    print("Target not found in file. Dump of file around expected area:")
    start_marker = "def __init__(self,"
    idx = content.find(start_marker)
    if idx != -1:
        print(content[idx:idx+300])
    else:
        print("Could not find __init__")
