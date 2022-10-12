# Installation

```bash
# Create virtual environment
python3 -m venv "venv"
# Activate virtual environment
source ./venv/bin/acticate
# Install pip tools
pip install pip-tools
# Compile dependencies
pip-compile
# Install dependencies
pip-sync
# Run program
python ./main.py
```