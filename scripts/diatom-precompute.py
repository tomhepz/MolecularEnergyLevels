# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: notebooks//ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# This file precomputes important values for use later
"""

# %% [markdown]
"""
# Import appropriate modules
"""

# %%
import numpy as np
from numpy.linalg import eigh
from numpy import save

import diatom.hamiltonian as hamiltonian
import diatom.calculate as calculate
from diatom.constants import Rb87Cs133

import scipy.constants

# %% [markdown]
"""
# Define 'global' Constants
"""

# %%
H_BAR = scipy.constants.hbar

I1 = Rb87Cs133["I1"]
I2 = Rb87Cs133["I2"]
D_0 = Rb87Cs133["d0"]
N_MAX=2

I = 0 #W/m^2
E = 0 #V/m

GAUSS = 1e4 # T
B_MIN = 0.01 / GAUSS # T
B_MAX = 600 / GAUSS # T
B_STEPS = 1200

B, B_STEP_SIZE = np.linspace(B_MIN, B_MAX, B_STEPS, retstep=True) #T 

# %% [markdown]
"""
# Build and Diagonalise Hamiltonian for many B
"""

# %%
H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(N_MAX, Rb87Cs133, zeeman=True, Edc=True, ac=True)

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I
H = H.transpose(2,0,1)

ENERGIES_UNSORTED, STATES_UNSORTED = eigh(H)
N_STATES = len(ENERGIES_UNSORTED[0])

# %%
# ENERGIES_HALF_SORTED, STATES_HALF_SORTED = calculate.sort_smooth(ENERGIES_UNSORTED,STATES_UNSORTED)
ENERGIES, STATES, LABELS = calculate.sort_by_state(ENERGIES_UNSORTED, STATES_UNSORTED, N_MAX, Rb87Cs133)

# %%
MAGNETIC_MOMENTS = calculate.magnetic_moment(STATES, N_MAX, Rb87Cs133)

# %%
dipole_op_zero = calculate.dipole(N_MAX,I1,I2,1,0)
dipole_op_minus = calculate.dipole(N_MAX,I1,I2,1,-1)
dipole_op_plus = calculate.dipole(N_MAX,I1,I2,1,+1)

COUPLINGS_ZERO = STATES[:, :, :].conj().transpose(0, 2, 1) @ (dipole_op_zero @ STATES[:, :, :])
COUPLINGS_MINUS = STATES[:, :, :].conj().transpose(0, 2, 1) @ (dipole_op_minus @ STATES[:, :, :])
COUPLINGS_PLUS = STATES[:, :, :].conj().transpose(0, 2, 1) @ (dipole_op_plus @ STATES[:, :, :])

COUPLINGS = [COUPLINGS_ZERO,COUPLINGS_PLUS,COUPLINGS_MINUS]

# %% [markdown]
"""
# Save to files
"""

# %%
save('../precomputed/energies-600G-1200Steps.npy', ENERGIES)
save('../precomputed/states-600G-1200Steps.npy', STATES)
save('../precomputed/labels-600G-1200Steps.npy', LABELS)
save('../precomputed/magnetic-moments-600G-1200Steps.npy', MAGNETIC_MOMENTS)

# %%
