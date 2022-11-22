# diatom-notebooks
A set of python notebooks for showcasing and explaining calculations involving hetronuclear diatomic molecules, specifically RbCs. Heavily uses the [diatom](https://github.com/PhilipDGregory/Diatomic-Py) library.

![energy level splitting diagram](/images/ZeemanRabi.png)  |  ![Animation of angular wavefunction with increasing E](/images/Animation.gif)
:---:|:---:

# Installation

```shell
# Create virtual environment
python3 -m venv "venv"
# Activate virtual environment
source ./venv/bin/acticate
# Install pip tools
pip install pip-tools
# Compile dependencies (implicitly from requirements.in)
pip-compile
# Install dependencies (implicitly from generated requirements.txt)
pip-sync

# Run as script
python ./scripts/raw-dc-stark.py

# ... OR open in jupyter
jupyter notebook
# because of jupytext plugin, can open scripts as notebook
# or % formatted python script

# To leave python venv environment
deactivate
```
