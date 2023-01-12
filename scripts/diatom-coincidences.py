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
# Import appropriate modules
"""

# %%
import numpy as np
from numpy.linalg import eigh
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})

import diatom.hamiltonian as hamiltonian
import diatom.calculate as calculate
from diatom.constants import Rb87Cs133

import scipy.constants

import matplotlib.pyplot as plt
import matplotlib.colors

import mpl_interactions.ipyplot as iplt
from mpl_interactions.controller import Controls

from functools import partial

import itertools

#plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = (4, 3.5)
plt.rcParams['figure.dpi'] = 200

# %matplotlib widget
# %config InlineBackend.figure_format = 'retina'

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
B_MAX = 350 / GAUSS # T
B_STEPS = 350

B, B_STEP_SIZE = np.linspace(B_MIN, B_MAX, B_STEPS, retstep=True) #T

PULSE_TIME = 100e-6 # s

def btoi(b):
    return (b-B_MIN)/B_STEP_SIZE

def itob(i):
    return B_STEP_SIZE*i+B_MIN


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
# Helper Functions
"""


# %%
def label_to_state_no(N,MF,k):
    for i, label in enumerate(LABELS):
        if label[0] == N and label[1] == MF and label[2] == k:
            return i

def state_no_to_uncoupled_label(state_no):
    i=0
    I1d = int(I1*2)
    I2d = int(I2*2)
    for n in range(0, N_MAX + 1):
        for mn in range(n,-(n+1),-1):
            for mi1 in range(I1d,-I1d-1,-2):
                for mi2 in range(I2d,-I2d-1,-2):
                    if i == state_no:
                        return (n,mn,mi1/2,mi2/2)
                    i+=1
    
print(label_to_state_no(*(2, 4, 13)))


# %%
def trio_transfer_efficiency(state1_label,state2_label,state3_label,bi,pulse_time=0.0001):
    state1i = label_to_state_no(*state1_label)
    state2i = label_to_state_no(*state2_label)
    state3i = label_to_state_no(*state3_label)
    
    P = state1_label[1] - state2_label[1]
    COUPLING = COUPLINGS[P]
    
    g = np.abs(COUPLING[bi, state1i, state3i]/COUPLING[bi, state1i, state2i])
    k = np.abs(((ENERGIES[bi, state3i] - ENERGIES[bi, state2i]) / scipy.constants.h) / (1/pulse_time))
    sub_transfered = ((1 + g**2)**2 + 8*(-1 + 2*g**2)*k**2 + 16*k**4) / ((1 + g**2)**3 + (-8 + 20*g**2 + g**4)*k**2 + 16*k**4)
    
    return sub_transfered


# %%
def transfer_efficiency(state1_label, state2_label,bi,pulse_time=0.0001):
    transfered = 1
    
    state1i = label_to_state_no(*state1_label)
    state2i = label_to_state_no(*state2_label)

    P = (state1_label[1] - state2_label[1])*(state2_label[0] - state1_label[0])
    COUPLING = COUPLINGS[P]
    
    for state3i in range(N_STATES):
        if state3i == state1i or state3i == state2i:
            continue
        g = np.abs(COUPLING[bi, state1i, state3i]/COUPLING[bi, state1i, state2i])
        k = np.abs(((ENERGIES[bi, state3i] - ENERGIES[bi, state2i]) / scipy.constants.h) / (1/pulse_time))
        sub_transfered = ((1 + g**2)**2 + 8*(-1 + 2*g**2)*k**2 + 16*k**4) / ((1 + g**2)**3 + (-8 + 20*g**2 + g**4)*k**2 + 16*k**4)
        transfered *= sub_transfered
        
    return transfered


# %%
print(transfer_efficiency((0,5,0),(1,4,1),int(B_STEPS/2)))
print(transfer_efficiency((1,4,1),(0,5,0),int(B_STEPS/2)))
print('----')
print(transfer_efficiency((0,5,0),(1,5,1),int(B_STEPS/2)))
print(transfer_efficiency((1,5,1),(0,5,0),int(B_STEPS/2)))
print('----')
print(transfer_efficiency((0,5,0),(1,6,0),int(B_STEPS/2)))
print(transfer_efficiency((1,6,0),(0,5,0),int(B_STEPS/2)))
print('---------')
print(transfer_efficiency((0,4,1),(1,3,1),int(B_STEPS/2)))
print(transfer_efficiency((1,3,1),(2,2,0),int(B_STEPS/2)))
print(transfer_efficiency((2,2,0),(1,3,0),int(B_STEPS/2)))
print(transfer_efficiency((1,3,0),(0,4,1),int(B_STEPS/2)))
print(transfer_efficiency((0,4,1),(1,3,0),int(B_STEPS/2)))

# %%
print(transfer_efficiency((0,5,0),(1,6,0),int(B_STEPS-1)))

eff = []
for bi in range(B_STEPS):
    eff.append(transfer_efficiency((0,5,0),(1,5,2),bi))
eff=np.array(eff)
fig,ax=plt.subplots()
ax.set_ylim(0,1.1)
ax.plot(B*GAUSS,eff)

# %% [markdown]
"""
# General Constants
"""

# %%
INITIAL_STATE_LABEL = (0,5,0)
INITIAL_STATE_POSITION = label_to_state_no(*INITIAL_STATE_LABEL)

# Ordered by energy low->high at 181.5G 
ACCESSIBLE_STATE_LABELS = [(1, 5, 0), (1, 4, 0), (1, 4, 1), (1, 6, 0), (1, 5, 1), (1, 4, 2), (1, 5, 2), (1, 4, 3), (1, 4, 4), (1, 4, 5)]
ACCESSIBLE_STATE_POSITIONS = [label_to_state_no(N,MF,k) for N,MF,k in ACCESSIBLE_STATE_LABELS]

CONSIDERED_STATE_LABELS = [INITIAL_STATE_LABEL] + ACCESSIBLE_STATE_LABELS
CONSIDERED_STATE_POSITIONS = [INITIAL_STATE_POSITION] + ACCESSIBLE_STATE_POSITIONS

STATE_CMAP = plt.cm.gist_rainbow(np.linspace(0,1,len(CONSIDERED_STATE_POSITIONS)))

# %% [markdown]
"""
# Debug sort function Find Magnetic Moments
"""

# %%
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
muN = scipy.constants.physical_constants['nuclear magneton'][0]

ax1.plot(B*GAUSS,MAGNETIC_MOMENTS[:,180:]/muN, color='grey', alpha=0.3,linewidth=0.5,zorder=0);
ax1.set_xlim(0,200)
ax1.set_ylabel("Magnetic Moment $\mu$ $(\mu_N)$")
ax1.set_xlabel("Magnetic Field $B_z$ (G)")

ax2.plot(B*GAUSS,ENERGIES[:,128:]/muN, color='grey', alpha=0.3,linewidth=0.5,zorder=0);
ax2.set_xlim(0,200)
ax2.set_ylabel("Energy")
ax2.set_xlabel("Magnetic Field $B_z$ (G)")



# %% [markdown]
"""
# Find Coindidences
"""


# %%
def label_degeneracy(N,MF):
    # Want number of ways of having
    # MF = MN + (M_I1 + M_I2) # NP-Hard Problem SSP (Subset Sum)
    d=0
    for MN in range(-N,N+1):
        for M_I1 in range(-int(I1+0.5),int(I1+0.5)+1):
            M_I1 = M_I1-0.5
            for M_I2 in range(-int(I2+0.5),int(I2+0.5)+1):
                M_I2 = M_I2-0.5
                if MN+M_I1+M_I2 == MF:
                    d+=1
    return d

print(label_degeneracy(0,5))
print(label_degeneracy(1,4))
print(label_degeneracy(1,5))
print(label_degeneracy(1,6))
print(label_degeneracy(1,7))

# %% tags=[]
# Find all possible combinations
base_mf = 4
polarisations = []
for p1 in [-1,0,1]:
    for p2 in [-1,0,1]:
        for p3 in [-1,0,1]:
            for p4 in [-1,0,1]:
                if p1+p2+p3+p4 == 0:
                    polarisations.append((p1,p2,p3,p4))

state_mfs = [(base_mf,base_mf+p1,base_mf+p1+p2,base_mf+p1+p2+p3) for p1,p2,p3,_ in polarisations]

states = []

for state_mf in state_mfs:
    for i in [1]:
        for j in range(label_degeneracy(1,state_mf[1])):
            for k in range(label_degeneracy(2,state_mf[2])):
                for l in range(label_degeneracy(1,state_mf[3])):
                    if (state_mf[1]<state_mf[3]) or (state_mf[1]==state_mf[3] and j<=l):
                        continue
                    states.append([(0,state_mf[0],i),(1,state_mf[1],j),(2,state_mf[2],k),(1,state_mf[3],l)])
                    
states=np.array(states)
                    
print(len(states), "states to consider")

# %%
# Find best B for minimum dipole deviation
best_deviation = np.ones((len(states)),dtype=np.double)
best_b_index = np.ones((len(states)),dtype=int)
for i,state_posibility in enumerate(states):
    desired_indices = np.array([
        label_to_state_no(*state_posibility[0]),
        label_to_state_no(*state_posibility[1]),
        label_to_state_no(*state_posibility[2]),
        label_to_state_no(*state_posibility[3])
    ])
    if desired_indices.all() != None:
        all_moments = MAGNETIC_MOMENTS[:,desired_indices]
        max_moment = np.amax(all_moments,axis=1)
        min_moment = np.amin(all_moments,axis=1)
        deviation = max_moment - min_moment
        
    min_diff_loc = np.argmin(deviation)
    min_diff = deviation[min_diff_loc]

    best_deviation[i] = np.abs(min_diff)
    best_b_index[i] = min_diff_loc

# order = best_deviation.argsort()

# where_interesting = np.array(where)[order]
# interesting_states = np.array(states)[order]
# interesting_deviation = np.array(ranking)[order]

# %%
# Simulate microwave transfers to find 'fidelity'
top_fidelities = np.zeros(len(states),dtype=np.double)
desired_pulse_time = 100 * 1e-6 # microseconds, longer => increased fidelity
for i, focus_state in enumerate(states):
    at_Bi = best_b_index[i]
    p = 1
    for fi in range(4):
        state_a_label = focus_state[fi%4]
        state_b_label = focus_state[(fi+1)%4]
        if (label_to_state_no(*state_a_label) is None) or (label_to_state_no(*state_b_label) is None):
            p*=0
        else:
            up_efficiency = transfer_efficiency(state_a_label,state_b_label,at_Bi,pulse_time=desired_pulse_time)
            down_efficiency = transfer_efficiency(state_b_label,state_a_label,at_Bi,pulse_time=desired_pulse_time)
            p *= up_efficiency * down_efficiency
        # print(p)
        # print(state_a_label,'->', state_b_label,'efficiency:', efficiency)
    top_fidelities[i] = p


# %%
# Rank state combinations
rating = np.zeros(len(states),dtype=np.double)
bestest_deviation = np.max(best_deviation)
bestest_fidelity = np.max(top_fidelities)
for i, focus_state in enumerate(states):
    deviation = bestest_deviation/best_deviation[i]
    fidelity = top_fidelities[i]
    rating[i] = deviation*fidelity

order = (-rating).argsort()

# %%
fig, axs = plt.subplots(4,4,figsize=(10,10),dpi=100,sharex=True,sharey=True,constrained_layout=True)

ordered_states = states[order]
ordered_B = best_b_index[order]
ordered_fidelities = top_fidelities[order]
ordered_deviations = best_deviation[order]

i=0
for axh in axs:
    for ax in axh:
        state_labels = ordered_states[i]
        state_numbers = np.array([label_to_state_no(*state_label) for state_label in ordered_states[i]])
        ax.set_xlim(0,B_MAX*GAUSS)
        ax.set_ylim(2,6)
        ax.set_xticks([0, 175, 350])
        ax.set_yticks([2, 4, 6])
        ax.plot(B*GAUSS,MAGNETIC_MOMENTS[:,state_numbers]/muN, alpha=1,linewidth=1.5,zorder=1);
        ax.axvline(x=min(B[ordered_B[i]]*GAUSS,B_MAX*GAUSS*0.98), dashes=(3, 2), color='k', linewidth=1.5,alpha=0.3,zorder=0)
        fidelity = ordered_fidelities[i]
        print(fidelity)
        max_dev = np.abs(ordered_deviations[i]/muN)
        ax.set_title(f'd={max_dev:.3f} f={fidelity:.3f}')
        ax.text(5,5.5,r'$|{},{}\rangle_{}$'.format(*(state_labels[0])))
        ax.text(85,5.5,r'$|{},{}\rangle_{}$'.format(*(state_labels[1])))
        ax.text(165,5.5,r'$|{},{}\rangle_{}$'.format(*(state_labels[2])))
        ax.text(245,5.5,r'$|{},{}\rangle_{}$'.format(*(state_labels[3])))
        i+=1

fig.supxlabel( 'Magnetic Field $B_z$ (G)')
fig.supylabel('Magnetic Moment $\mu$ $(\mu_N)$')

# fig.savefig('../images/magnetic-dipole-coincides.pdf')

# %%
