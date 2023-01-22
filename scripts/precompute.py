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
# This file computes diagonalisation & pre-computes results for use later
"""

# %% [markdown]
"""
## Import appropriate modules
"""

# %%
import numpy as np
from numpy.linalg import eigh
from numpy import save, savez, savez_compressed
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import diatom.hamiltonian as hamiltonian
import diatom.calculate as calculate
from diatom.constants import Rb87Cs133

import scipy.constants
from scipy.sparse import csgraph

# %% [markdown]
"""
## Define parameters
"""

# %%
H_BAR = scipy.constants.hbar

I1 = Rb87Cs133["I1"]
I2 = Rb87Cs133["I2"]
D_0 = Rb87Cs133["d0"]
N_MAX=2

I = 0 #W/m^2
E = 0 #V/m

B_MIN_GAUSS = 0.001
B_MAX_GAUSS = 1000
B_STEPS = 500
PULSE_TIME_US = 500

settings_string = f'NMax{N_MAX}BMin{B_MIN_GAUSS}BMax{B_MAX_GAUSS}BSteps{B_STEPS}PTime{PULSE_TIME_US}'
print(settings_string)

GAUSS = 1e4 # T
B_MIN = B_MIN_GAUSS / GAUSS # T
B_MAX = B_MAX_GAUSS / GAUSS # T
PULSE_TIME = PULSE_TIME_US / 1e6

B, B_STEP_SIZE = np.linspace(B_MIN, B_MAX, B_STEPS, retstep=True) #T 

# %% [markdown]
"""
## Diagonalise & Calculate 
"""

# %%
H0,Hz,Hdc,Hac = hamiltonian.build_hamiltonians(N_MAX, Rb87Cs133, zeeman=True, Edc=True, ac=True)
N_STATES = len(H0)

H = H0[..., None]+\
    Hz[..., None]*B+\
    Hdc[..., None]*E+\
    Hac[..., None]*I
H = H.transpose(2,0,1)

# %%
ENERGIES_UNSORTED, STATES_UNSORTED = eigh(H)

# %%
ENERGIES_HALF_SORTED, STATES_HALF_SORTED = calculate.sort_smooth(ENERGIES_UNSORTED,STATES_UNSORTED)

# %%
ENERGIES, STATES, LABELS = calculate.sort_by_state(ENERGIES_HALF_SORTED, STATES_HALF_SORTED, N_MAX, Rb87Cs133)

# %%
LABELS=(np.rint(LABELS)).astype(int)

# %%
MAGNETIC_MOMENTS = np.einsum('bji,jk,bki->bi', STATES.conj(), -Hz, STATES, optimize='optimal')

# %%
dipole_op_zero = calculate.dipole(N_MAX,I1,I2,1,0)
dipole_op_minus = calculate.dipole(N_MAX,I1,I2,1,-1)
dipole_op_plus = calculate.dipole(N_MAX,I1,I2,1,+1)

# %%
COUPLINGS_ZERO = STATES[:, :, :].conj().transpose(0, 2, 1) @ dipole_op_zero @ STATES[:, :, :]
COUPLINGS_MINUS = STATES[:, :, :].conj().transpose(0, 2, 1) @ dipole_op_minus @ STATES[:, :, :]
COUPLINGS_PLUS = STATES[:, :, :].conj().transpose(0, 2, 1) @ dipole_op_plus @ STATES[:, :, :]
COUPLINGS_UNPOLARISED = COUPLINGS_ZERO + COUPLINGS_MINUS + COUPLINGS_PLUS

# %% [markdown]
"""
# All pairs best fidelity
"""


# %%
def twice_average_fidelity(k,g):
    return ((1 + g**2)**2 + 8*k**2*(-1 + 2*g**2) + 16*k**4)/((1 + g**2)**3 + (-8 + 20*g**2 + g**4)*k**2 + 16*k**4)


# %%
POLARISED_PAIR_FIDELITIES = np.zeros((N_STATES,N_STATES,B_STEPS))
UNPOLARISED_PAIR_FIDELITIES = np.zeros((N_STATES,N_STATES,B_STEPS))

for Na in range(N_MAX):
    Nb = Na+1
    lowera = sum(32*(2*N+1) for N in range(Na))
    uppera = lowera + 32*(2*Na+1)
    upperb = uppera + 32*(2*Nb+1)
    for i in range(lowera,uppera):
        li = LABELS[i,:]
        for j in range(uppera,upperb):
            lj = LABELS[j,:]
            P = round((lj[1]-li[1])*(li[0]-lj[0]))
            if abs(P) >= 2:
                continue

            couplings_polarised = [COUPLINGS_ZERO,COUPLINGS_PLUS,COUPLINGS_MINUS][P]

            ks_up = np.abs((ENERGIES[:, :] - ENERGIES[:, j, None]) * PULSE_TIME / scipy.constants.h)
            ks_down = np.abs((ENERGIES[:, :] - ENERGIES[:, i, None]) * PULSE_TIME / scipy.constants.h)

            gs_unpolarised_up = np.abs(COUPLINGS_UNPOLARISED[:, i, :]/COUPLINGS_UNPOLARISED[:, i, j, None])
            gs_polarised_up = np.abs(couplings_polarised[:, i, :]/couplings_polarised[:, i, j, None])

            gs_unpolarised_down = np.abs(COUPLINGS_UNPOLARISED[:, j, :]/COUPLINGS_UNPOLARISED[:, i, j, None])
            gs_polarised_down = np.abs(couplings_polarised[:, j, :]/couplings_polarised[:, i, j, None])

            fidelities_unpolarised_up = twice_average_fidelity(ks_up,gs_unpolarised_up)
            fidelities_unpolarised_down = twice_average_fidelity(ks_down,gs_unpolarised_down)

            fidelities_polarised_up = twice_average_fidelity(ks_up,gs_polarised_up)
            fidelities_polarised_down = twice_average_fidelity(ks_down,gs_polarised_down)

            fidelities_unpolarised_up[:,np.array([i,j])] = 1
            fidelities_unpolarised_down[:,np.array([i,j])] = 1
            fidelities_polarised_up[:,np.array([i,j])] = 1
            fidelities_polarised_down[:,np.array([i,j])] = 1

            UNPOLARISED_PAIR_FIDELITIES[i,j,:] = np.prod(fidelities_unpolarised_up,axis=1) * np.prod(fidelities_unpolarised_down,axis=1)
            POLARISED_PAIR_FIDELITIES[i,j,:] = np.prod(fidelities_polarised_up,axis=1) * np.prod(fidelities_polarised_down,axis=1)

# %% [markdown]
"""
# Save to files
"""

# %%
np.savez_compressed(f'../precomputed/{settings_string}.npz', 
                    energies=ENERGIES,
                    states=STATES, 
                    labels=LABELS, 
                    magnetic_moments=MAGNETIC_MOMENTS, 
                    coupling_matrix_zero = COUPLINGS_ZERO,
                    coupling_matrix_minus = COUPLINGS_MINUS,
                    coupling_matrix_plus = COUPLINGS_PLUS,
                    unpolarised_pair_fidelities = UNPOLARISED_PAIR_FIDELITIES,
                    polarised_pair_fidelities = POLARISED_PAIR_FIDELITIES
                   )

# %% [markdown]
"""
# How to load file
"""

# %%
data = np.load(f'../precomputed/{settings_string}.npz')
energies_loaded = data['energies']
print(energies_loaded.shape)
