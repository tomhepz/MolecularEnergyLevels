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
from numpy import load
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})

import diatom.hamiltonian as hamiltonian
import diatom.calculate as calculate
from diatom.constants import Rb87Cs133

import scipy.constants
from scipy.linalg import expm

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec

from tqdm import tqdm, trange
import math
import itertools

from numba import jit, njit
from numba import njit
from numba_progress import ProgressBar

# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = 'sans-serif'
plt.rcParams["figure.autolayout"] = True
plt.rcParams['figure.figsize'] = (4, 3.5)
plt.rcParams['figure.dpi'] = 200
# plt.rc('text.latex', preamble=r'\usepackage[T1]{fontenc}\usepackage{cmbright}\usepackage{mathtools}')

# %matplotlib widget
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
"""
## Defining parameters
"""

# %%
MOLECULE_STRING = "Rb87Cs133"
MOLECULE = Rb87Cs133
N_MAX=2

settings_string = f'{MOLECULE_STRING}NMax{N_MAX}'
print(settings_string)

H_BAR = scipy.constants.hbar
muN = scipy.constants.physical_constants['nuclear magneton'][0]

I1 = MOLECULE["I1"]
I2 = MOLECULE["I2"]
I1_D = round(2*I1)
I2_D = round(2*I2)

D_0 = MOLECULE["d0"]

PER_MN = round((2*I1+1)*(2*I2+1))
N_STATES = PER_MN * (N_MAX+1)**2
F_D_MAX = 2*N_MAX + I1_D + I2_D
print(f"{N_STATES} states loaded from molecule.")

GAUSS = 1e-4 # T

# %% [markdown]
"""
# Load precomputed results
"""

# %%
print("Loading precomputed data...")
data = np.load(f'../precomputed/{settings_string}.npz')

B=data['b']
B_MIN = B[0]
B_MAX = B[-1]
B_STEPS = len(B)

ENERGIES = data['energies']
STATES = data['states']

UNCOUPLED_LABELS_D=data['uncoupled_labels_d']

LABELS_D=data['labels_d']
LABELS_DEGENERACY = data['labels_degeneracy']
STATE_JUMP_LIST = data['state_jump_list']

TRANSITION_LABELS_D = data['transition_labels_d']
TRANSITION_INDICES = data['transition_indices']
EDGE_JUMP_LIST = data['edge_jump_list']


MAGNETIC_MOMENTS=data['magnetic_moments'] 

COUPLINGS_SPARSE=data['couplings_sparse']
TRANSITION_GATE_TIMES_POL = data['transition_gate_times_pol']
TRANSITION_GATE_TIMES_UNPOL = data['transition_gate_times_unpol']

CUMULATIVE_TIME_FROM_INITIALS_POL = data['cumulative_pol_time_from_initials']
PREDECESSOR_POL = data['predecessor_pol_time_from_initials']

CUMULATIVE_TIME_FROM_INITIALS_UNPOL = data['cumulative_unpol_time_from_initials']
PREDECESSOR_UNPOL = data['predecessor_unpol_time_from_initials']

PAIR_RESONANCE = data['pair_resonance']

def label_degeneracy(N,MF_D):
    return LABELS_DEGENERACY[N,(MF_D+F_D_MAX)//2]

@jit(nopython=True)
def label_d_to_node_index(N,MF_D,d):
    return STATE_JUMP_LIST[N,(MF_D+F_D_MAX)//2]+d

@jit(nopython=True)
def label_d_to_edge_indices(N,MF_D,d): # Returns the start indices of P=0,P=1,P=2, and the next edge
    return EDGE_JUMP_LIST[label_d_to_node_index(N,MF_D,d)]

INITIAL_STATE_LABELS_D = MOLECULE["StartStates_D"]
INITIAL_STATE_INDICES = np.array([label_d_to_node_index(*label_d) for label_d in INITIAL_STATE_LABELS_D])
N_INITIAL_STATES = len(INITIAL_STATE_INDICES)
print("Loaded precomputed data.")

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
"""
# Helper Functions
"""


# %%
def label_d_to_string(label_d):
    n = label_d[0]
    mf_d = label_d[1]
    mf_frac = mf_d%2
    i = label_d[2]
    if mf_frac == 0:
        return f"({n},{mf_d//2},{i})"
    else:
        mf_whole= (mf_d - np.sign(mf_d))//2
        return f"({n},{mf_whole}.5,{i})"

def label_d_to_latex_string(label_d):
    n = label_d[0]
    mf_d = label_d[1]
    mf_frac = mf_d%2
    i = label_d[2]
    if mf_frac == 0:
        return r'|{},{}\rangle_{{{}}}'.format(n,mf_d//2,i)
    else:
        mf_whole= (mf_d - np.sign(mf_d))//2
        return r'|{},{}.5\rangle_{{{}}}'.format(n,mf_whole,i)
    
def fid_to_string(fid):
    return f"{fid:.4f}({fidelity(fid,d=9):.1f})"


# %%
def reachable_above_from(N,MF_D):
    sigma_plus_reachable = [(N+1,MF_D-2,i) for i in range(label_degeneracy(N+1,MF_D-2))]
    pi_reachable = [(N+1,MF_D,i) for i in range(label_degeneracy(N+1,MF_D))]
    sigma_minus_reachable = [(N+1,MF_D+2,i) for i in range(label_degeneracy(N+1,MF_D+2))]
    return (sigma_plus_reachable + pi_reachable + sigma_minus_reachable)

def twice_average_fidelity(k,g):
    return ((1 + g**2)**2 + 8*k**2*(-1 + 2*g**2) + 16*k**4)/((1 + g**2)**3 + (-8 + 20*g**2 + g**4)*k**2 + 16*k**4)

def simple_fidelity(k,g):
    g2 = g**2
    k2 = k**2
    return  1 - (4*g2+g2**2)/(16*k2)

def maximum_fidelity(k,g):
    phi = np.arccos((k*(18-9*g**2-8*k**2))/(3+3*g**2+4*k**2)**(3/2))/3
    denominator = 54*((1+g**2)**3+(-8+20*g**2+g**4)*k**2+16*k**4)
    numerator = (
                 36*(g**4+(1-4*k**2)**2+2*g**2*(1+8*k**2))
               + 32*k    *(3+3*g**2+4*k**2)**(3/2) *np.cos(phi)
               - 64*k**2 *(3+3*g**2+4*k**2)        *np.cos(2*phi) 
               -  4      *(3+3*g**2+4*k**2)**2     *np.cos(4*phi)
                )
    return numerator/denominator

def fidelity(ts,d=8):
    return -np.log10(1-ts+10**(-d))


# %%
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def field_to_bi(gauss):
    return find_nearest(B,gauss*GAUSS)


# %%
@jit(nopython=True)
def label_pair_to_edge_index(label1,label2):
    first_indices = label_d_to_edge_indices(label1[0],label1[1],label1[2])
    section = 3*((label2[0] - label1[0]) < 0) + [0,1,2][(label2[0] - label1[0])*(label1[1] - label2[1])//2]
    return first_indices[section]+label2[2]

# TRANSITION_LABELS_D[label_pair_to_edge_index((1,4,3),(0,2,1))]
# TRANSITION_LABELS_D[label_pair_to_edge_index(np.array([1,4,3]),np.array([0,2,1]))]


# %% [markdown] tags=[]
"""
# Simulator
"""


# %%
def simulate(chosen_states_coupling_labels, chosen_coupling_labels, chosen_pulse_time, chosen_bi, initial_populations, T_STEPS=41443, resolution=1):

    # Simulation time length (how many Rabi periods to show)
    TIME = chosen_pulse_time[0]*2

    chosen_state_labels = []
    # example_points = []
    POLARISED = False

    needed_states = np.full(N_STATES, False)
    for l in chosen_states_coupling_labels:
        ni = label_d_to_edge_indices(*l)
        nl = TRANSITION_INDICES[ni[0]:ni[6]][:,1]
        needed_states[nl] = True 
        # example_points.append()

    # chosen_states_labels = LABELS_D
    chosen_states_labels = LABELS_D[needed_states]
    # chosen_states_labels = LABELS_D[np.array([label_d_to_node_index(*label) for label in np.concatenate((chosen_states_coupling_labels,np.array([[0,10,0]])))])]
    # print(chosen_states_labels)

    chosen_states_coupling_subindices = [np.where((chosen_states_labels[:, 0] == N) & (chosen_states_labels[:, 1] == MF) & (chosen_states_labels[:, 2] == k))[0][0] for N,MF,k in chosen_states_coupling_labels]
    chosen_states_indices = np.array([label_d_to_node_index(*label) for label in chosen_states_labels])
    chosen_number_of_states = len(chosen_states_indices)
    
    
    # Get Angular Frequency Matrix Diagonal for each B
    all_angular = ENERGIES[:, chosen_bi].real / H_BAR # [state]
    angular = all_angular[chosen_states_indices]

    # Form coupling matrix
    couplings = np.zeros((chosen_number_of_states,chosen_number_of_states))
    for i in range(chosen_number_of_states):
        ai = chosen_states_indices[i]
        for j in range(chosen_number_of_states):
            bi = chosen_states_indices[j]
            la = LABELS_D[ai]
            lb = LABELS_D[bi]
            if abs(la[0]-lb[0]) != 1 or abs(la[1]-lb[1]) > 2:
                continue
            edge_index = label_pair_to_edge_index(la,lb)
            couplings[i,j] = COUPLINGS_SPARSE[edge_index,chosen_bi]

    # Get driving frequencies & polarisations
    driving = []
    E_i = []
    for (l1,l2), pulse_time in zip(chosen_coupling_labels, chosen_pulse_time):
        i1=label_d_to_node_index(*l1)
        i2=label_d_to_node_index(*l2)
        driving.append(np.abs(all_angular[i1]-all_angular[i2]))
        E_i.append((2*np.pi*H_BAR) / (D_0 * COUPLINGS_SPARSE[label_pair_to_edge_index(l1,l2),chosen_bi] * pulse_time))
    driving = np.array(driving)
    E_i = np.array(E_i,dtype=np.double)

    # Construct times
    times, DT = np.linspace(0,TIME,num=T_STEPS,retstep=True)

    # Construct kinetic time step operator (Matrix Diagonal)
    T_OP_DIAG = np.exp(-(1j) * angular * DT/2 )

    # Construct potential fixed part time step operator 
    ORDER = 10
    V_TI_M = (-(1j)*D_0*couplings*DT)/H_BAR
    V_TI_M_POWS = np.array([np.linalg.matrix_power(V_TI_M, i)/np.math.factorial(i) for i in range(ORDER)])

    # Construct state vector
    state_vector = np.zeros((T_STEPS,chosen_number_of_states), dtype=np.cdouble)
    for i,p in enumerate(initial_populations):
        state_vector[0,chosen_states_coupling_subindices[i]] = np.sqrt(p)

    for t_num in trange(T_STEPS-1):
        pres = E_i*np.cos(driving*times[t_num])
        V_TD_POWS = np.sum(pres)**(np.arange(ORDER))
        V_OP = np.sum(V_TI_M_POWS*V_TD_POWS[:,None,None],axis=0)

        DU = T_OP_DIAG[:,None] * V_OP[:,:] * T_OP_DIAG[None,:]
        state_vector[t_num+1] = DU @ state_vector[t_num]
    
    return times[::resolution], np.abs(state_vector[::resolution,:])**2, chosen_states_coupling_subindices

# %%
# Driven couplings between states
chosen_states_coupling_labels = np.array([(0,8,1),(1,8,3),(2,8,3),(1,6,2)])
chosen_coupling_labels = [
    (chosen_states_coupling_labels[0],chosen_states_coupling_labels[1]),
    (chosen_states_coupling_labels[1],chosen_states_coupling_labels[2]),
    (chosen_states_coupling_labels[3],chosen_states_coupling_labels[2]),
    (chosen_states_coupling_labels[0],chosen_states_coupling_labels[3]),
]

# With what desired rabi period
f_boost = 0

global_pulse_time = 1100 * 2 * 1e-6 * 10**(f_boost/2) #s
chosen_pulse_time = [global_pulse_time]*len(chosen_coupling_labels)

# At what magnetic field
chosen_bi = field_to_bi(710)

# Simulation resolution
T_STEPS =  [803989,314927,195931,65519,41443,21319,9391,50][4]*2

times, probabilities, chosen_states_coupling_subindices = simulate(chosen_states_coupling_labels,chosen_coupling_labels,chosen_pulse_time,chosen_bi,[0,1,0],T_STEPS)

# %%
# Plot results
fig,ax = plt.subplots(figsize=(6.5,2.5),zorder=-2,constrained_layout=True)
ax.set_xlabel('$t\,(\mu s)$')
ax.set_ylabel('$P_i$')
ax.set_ylim(0,1.0)
ax.set_xlim(0,global_pulse_time*2*1e6)
# ax.set_xlim(0,5000)
states_string = ','.join([f'${label_d_to_latex_string(label)}$' for label in chosen_states_coupling_labels])
# ax.set_title('{},\n {} @ {}G,\n RabiPeriod {}$\mu s$, SimSteps {},\n unpolarised, {} states simulated'
#              .format(MOLECULE_STRING,
#                      states_string,
#                      f"{B[chosen_bi]/GAUSS:.1f}",
#                      round(global_pulse_time*1e6),
#                      T_STEPS,
#                      chosen_number_of_states))

c = ['blue','green','purple','red','grey','grey']
# ax.plot(times*1e6,probabilities[:,:],c='grey',linewidth=0.5,alpha=0.5,zorder=-2);
for i,state_subindex in enumerate(chosen_states_coupling_subindices):
    ax.plot(times*1e6,probabilities[:,state_subindex],c=c[i],linewidth=1.5,zorder=-2);
    
inset_gap = 10**(-f_boost-3)

# Create a GridSpec for the inset axes
inset_gs = GridSpec(2, 1, left=0.58, bottom=0.35, right=0.9, top=0.85, hspace=0.1)

# Create the inset axes using the GridSpec
axinset = fig.add_subplot(inset_gs[0, 0],zorder=10)
axinset2 = fig.add_subplot(inset_gs[1, 0], sharex=axinset,zorder=10)
plt.setp(axinset.get_xticklabels(), visible=False)
    
# axinset = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
# axinset.plot(times*1e6,probabilities[:,:],c='grey',linewidth=0.5,alpha=0.7);
for i,state_subindex in enumerate(chosen_states_coupling_subindices):
    axinset.plot(times*1e6,probabilities[:,state_subindex],c=c[i],linewidth=1.0);
axinset.set_xlim(0,global_pulse_time*2*1e6)
axinset.set_ylim(1-3*inset_gap,1)
axinset.axhline(1-inset_gap,color='black',linestyle='dashed',lw=1)

# axinset2 = ax.inset_axes([0.6, 0.13, 0.3, 0.3])
axinset2.plot(times*1e6,probabilities[:,:],c='grey',linewidth=0.5,alpha=0.7);
for i,state_subindex in enumerate(chosen_states_coupling_subindices):
    axinset2.plot(times*1e6,probabilities[:,state_subindex],c=c[i],linewidth=1.0);
axinset2.set_xlim(0,global_pulse_time*2*1e6)
axinset2.set_ylim(0,3*inset_gap)
axinset2.axhline(inset_gap,color='black',linestyle='dashed',lw=1)


# Split axes formatting
axinset.spines.bottom.set_visible(False)
axinset2.spines.top.set_visible(False)
axinset.xaxis.tick_top()
axinset.tick_params(labeltop=False)
axinset2.xaxis.tick_bottom()
d = .2  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axinset.plot([0, 1], [0, 0], transform=axinset.transAxes, **kwargs)
axinset2.plot([0, 1], [1, 1], transform=axinset2.transAxes, **kwargs)

rect = matplotlib.patches.Rectangle((0.58, 0.35), 0.3, 0.5, facecolor='white', edgecolor='none', transform=fig.transFigure, figure=fig,zorder=0)
fig.patches.extend([rect])


axinset3 = ax.inset_axes([0.12, 0.4, 0.3, 0.3])
# axinset2.plot(times*1e6,probabilities[:,:],c='grey',linewidth=0.5,alpha=0.5);
for i,state_subindex in enumerate(chosen_states_coupling_subindices):
    axinset3.plot(times*1e6,probabilities[:,state_subindex],c=c[i],linewidth=1.0);
axinset3.set_xlim(500,600)
axinset3.set_ylim(0.25*0.99,0.25*1.01)
axinset3.axhline(inset_gap,color='black',linestyle='dashed',lw=1)
axinset3.axhline(0.25+0.001,color='black',linestyle='dashed',lw=1)
axinset3.axhline(0.25-0.001,color='black',linestyle='dashed',lw=1)
    
print(f"{np.max(probabilities[:,chosen_states_coupling_subindices[1]]):.10f}")
fig.savefig(f'../images/{MOLECULE_STRING}-4-state-loop-sim.pdf')

# %%
chosen_coupling_labels = [(chosen_states_coupling_labels[0],chosen_states_coupling_labels[1])]
times_a, probabilities_a, chosen_states_coupling_subindices = simulate(chosen_states_coupling_labels,chosen_coupling_labels,chosen_pulse_time,chosen_bi,[0,1,0])

chosen_coupling_labels = [(chosen_states_coupling_labels[2],chosen_states_coupling_labels[1])]
times_b, probabilities_b, chosen_states_coupling_subindices = simulate(chosen_states_coupling_labels,chosen_coupling_labels,chosen_pulse_time,chosen_bi,[0,1,0])

chosen_coupling_labels = [(chosen_states_coupling_labels[0],chosen_states_coupling_labels[1])]
times_c, probabilities_c, chosen_states_coupling_subindices = simulate(chosen_states_coupling_labels,chosen_coupling_labels,chosen_pulse_time,chosen_bi,[0,0,1])

chosen_coupling_labels = [(chosen_states_coupling_labels[1],chosen_states_coupling_labels[2])]
times_d, probabilities_d, chosen_states_coupling_subindices = simulate(chosen_states_coupling_labels,chosen_coupling_labels,chosen_pulse_time,chosen_bi,[1,0,0])

# %%
# Plot results
fig, axs = plt.subplots(4,2,figsize=(6.2,2.5),constrained_layout=True,sharex=True)
# ax_a.set_xlabel('$t\,(\mu s)$')
# ax.set_ylabel('$P_i$')
# ax.set_ylim(0,1.0)
# ax.set_xlim(0,TIME*1e6)
# ax.set_xlim(0,2000)
# states_string = ','.join([f'${label_d_to_latex_string(label)}$' for label in chosen_states_coupling_labels])
# ax.set_title('{},\n {} @ {}G,\n RabiPeriod {}$\mu s$, SimSteps {},\n unpolarised, {} states simulated'
#              .format(MOLECULE_STRING,
#                      states_string,
#                      f"{B[chosen_bi]/GAUSS:.1f}",
#                      round(global_pulse_time*1e6),
#                      T_STEPS,
#                      chosen_number_of_states))
# print(axs.shape)

axs[3,0].set_xlabel("time ($\mu s$)")
axs[3,1].set_xlabel("time ($\mu s$)")

axs[1,0].set_ylabel("Populations           ")
for axl,axh,times,probs,label,xp,yp in [(axs[1,0],axs[0,0],times_a,probabilities_a,r"$|e\rangle\leftrightarrow |0\rangle,\,|e\rangle$ populated",0,0),(axs[3,0],axs[2,0],times_b,probabilities_b,r"$|e\rangle\leftrightarrow |1\rangle,\,|e\rangle$ populated",0,1),(axs[1,1],axs[0,1],times_c,probabilities_c,r"$|e\rangle\leftrightarrow |0\rangle,\,|1\rangle$ populated",1,0),(axs[3,1],axs[2,1],times_d,probabilities_d,r"$|e\rangle\leftrightarrow |1\rangle,\,|0\rangle$ populated",1,1)]:
    # ax.set_ylim(0,1.0)
    # # ax.set_xlim(0,TIME*1e6)
    # ax.set_xlim(0,2000)
    
    
    c = ['blue','green','purple','red','grey','grey']
    # ax.plot(times*1e6,probs[:,:],c='grey',linewidth=1.5,alpha=0.5);
    # for i,state_subindex in enumerate(chosen_states_coupling_subindices):
    #     ax.plot(times*1e6,probs[:,state_subindex],c=c[i],linewidth=1.5);
        
    inset_line = 0.001
    inset_drop = 0.002

    # axh = ax.inset_axes([0.0, 0.6, 1.0, 0.4])
    axh.plot(times*1e6,probs[:,:],c='grey',linewidth=1.5,alpha=0.5);
    for i,state_subindex in enumerate(chosen_states_coupling_subindices):
        axh.plot(times*1e6,probs[:,state_subindex],c=c[i],linewidth=1.5);
    axh.set_xlim(0,2000)
    axh.set_ylim(1-inset_drop,1)
    axh.axhline(1-inset_line,color='black',linestyle='dashed',lw=1)
    axh.axhline(0.001)
    axh.set_yticks([1-inset_line,1])

    # axl = ax.inset_axes([0.0, 0.0, 1.0, 0.4])
    axl.plot(times*1e6,probs[:,:],c='grey',linewidth=1.5,alpha=0.5);
    for i,state_subindex in enumerate(chosen_states_coupling_subindices):
        axl.plot(times*1e6,probs[:,state_subindex],c=c[i],linewidth=1.5);
    axl.set_xlim(0,2000)
    axl.set_ylim(0,inset_drop)
    axl.axhline(inset_line,color='black',linestyle='dashed',lw=1)
    axl.set_yticks([0,inset_line])
    
    # Label
    fig.text(0.297+0.485*xp, 0.36+0.43*yp, label, ha='center', va='center')
    
    # Split axes formatting
    axh.spines.bottom.set_visible(False)
    axl.spines.top.set_visible(False)
    axh.xaxis.tick_top()
    axh.tick_params(labeltop=False)
    axl.xaxis.tick_bottom()
    d = .2  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axh.plot([0, 1], [0, 0], transform=axh.transAxes, **kwargs)
    axl.plot([0, 1], [1, 1], transform=axl.transAxes, **kwargs)


    
# print(f"{np.max(probabilities[:,chosen_states_coupling_subindices[1]]):.10f}")
fig.savefig(f'../images/{MOLECULE_STRING}-2-state-qubit-sim.pdf')

# %%
