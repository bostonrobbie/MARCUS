file_path = r"C:\Users\User\Documents\AI\examples\nqorb_enhanced.py"

with open(file_path, 'r') as f:
    content = f.read()

# Replace all occurrences of rvol > 3.0 with rvol > 2.0
if "if rvol > 3.0:" in content:
    new_content = content.replace("if rvol > 3.0:", "if rvol > 2.0:")
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully patched nqorb_enhanced.py")
else:
    print("Target string 'if rvol > 3.0:' not found.")
