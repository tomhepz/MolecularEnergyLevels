# What is this?

This code numerically computes the energy levels for a hetronuclear molecule modeled as a rigid rotor within in a DC electric field.

![Animation of angular wavefunction with increasing E](/images/Animation.gif)

![energy level splitting diagram](/images/DCStarkRigidRotor.png)

# Installation

```shell
# Create virtual environment
python3 -m venv "venv"
# Activate virtual environment
source ./venv/bin/acticate
# Install pip tools
pip install pip-tools
# Compile dependencies
pip-compile requirements.in
pip-compile dev-requirements.in
# Install dependencies
pip-sync dev-requirements.txt

# Run as script
python ./dc-stark.py

# ... OR open in jupyter
jupyter lab
# because of jupytext plugin, can open learning.py as notebook

# To leave python venv environment
deactivate
```
