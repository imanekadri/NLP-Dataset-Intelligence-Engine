import os

target_emoji = "\U0001f680"

for root, dirs, files in os.walk("."):
    if "venv" in root or ".git" in root or "output" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if target_emoji in line:
                            print(f"FOUND in {path} at line {i+1}")
            except:
                continue
