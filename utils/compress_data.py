import zipfile
import os
import glob
from pathlib import Path

def compress_data():
    # Search recursively for csv files
    root_dir = Path(".")
    data_files = list(root_dir.rglob("A2API*.csv"))
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    for file in data_files:
        # Create zip in the 'data' folder
        zip_name = f"data/{file.name}.zip"
        print(f"Compressing {file} to {zip_name}...")
        
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(file, arcname=file.name)
            
        size_mb = os.path.getsize(zip_name) / (1024 * 1024)
        print(f"Created {zip_name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    compress_data()
