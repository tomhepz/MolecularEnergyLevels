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
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3e}".format(x)})

import diatom.hamiltonian as hamiltonian
import diatom.calculate as calculate
from diatom.constants import *

import scipy.constants
from scipy.sparse import csgraph

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec

import mpl_interactions.ipyplot as iplt
from mpl_interactions.controller import Controls

from functools import partial

import itertools

from tabulate import tabulate

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
MOLECULE_STRING = "Na23K40"
MOLECULE = Na23K40
N_MAX=2

B_MIN_GAUSS = 0.001 #G
B_MAX_GAUSS = 1000 #G
B_STEPS = 500

PULSE_TIME_US = 500 #μs

settings_string = f'{MOLECULE_STRING}NMax{N_MAX}BMin{B_MIN_GAUSS}BMax{B_MAX_GAUSS}BSteps{B_STEPS}PTime{PULSE_TIME_US}'

# %% [markdown]
"""
## Computed Constants
"""

# %%
H_BAR = scipy.constants.hbar
muN = scipy.constants.physical_constants['nuclear magneton'][0]

I1 = MOLECULE["I1"]
I2 = MOLECULE["I2"]
I1_D = round(2*I1)
I2_D = round(2*I2)

D_0 = MOLECULE["d0"]

PER_MN = round((2*I1+1)*(2*I2+1))
N_STATES = PER_MN * (N_MAX+1)**2

GAUSS = 1e-4 # T
B_MIN = B_MIN_GAUSS * GAUSS # T
B_MAX = B_MAX_GAUSS * GAUSS # T
PULSE_TIME = PULSE_TIME_US * 1e-6 # s

B, B_STEP_SIZE = np.linspace(B_MIN, B_MAX, B_STEPS, retstep=True) #T 

# %% [markdown]
"""
# Load precomputed results
"""

# %%
data = np.load(f'../precomputed/{settings_string}.npz')

ENERGIES = data['energies']
STATES = data['states']

LABELS_D=data['labels_d']
UNCOUPLED_LABELS_D=data['uncoupled_labels_d']

MAGNETIC_MOMENTS=data['magnetic_moments'] 

COUPLINGS_ZERO=data['couplings_zero']
COUPLINGS_MINUS=data['couplings_minus']
COUPLINGS_PLUS=data['couplings_plus']
COUPLINGS = COUPLINGS_ZERO+COUPLINGS_MINUS+COUPLINGS_PLUS
POLARISED_COUPLING = [COUPLINGS_ZERO,COUPLINGS_PLUS,COUPLINGS_MINUS]

UNPOLARISED_PAIR_FIDELITIES = data['unpolarised_pair_fidelities_ut']
UNPOLARISED_PAIR_FIDELITIES = UNPOLARISED_PAIR_FIDELITIES + UNPOLARISED_PAIR_FIDELITIES.transpose(1,0,2)
POLARISED_PAIR_FIDELITIES = data['polarised_pair_fidelities_ut']
POLARISED_PAIR_FIDELITIES = POLARISED_PAIR_FIDELITIES + POLARISED_PAIR_FIDELITIES.transpose(1,0,2)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
"""
# Helper Functions
"""


# %%
def label_to_state_no(N,MF_D,k):
    return np.where((LABELS_D[:, 0] == N) & (LABELS_D[:, 1] == MF_D) & (LABELS_D[:, 2] == k))[0][0]

def state_no_to_uncoupled_label(state_no):
    return UNCOUPLED_LABELS_D[state_no]


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
        return r'$|{},{}\rangle_{{{}}}$'.format(n,mf_d//2,i)
    else:
        mf_whole= (mf_d - np.sign(mf_d))//2
        return r'$|{},{}\frac{{1}}{{2}}\rangle_{{{}}}$'.format(n,mf_whole,i)


# %%
def label_degeneracy(N,MF_D):
    # Want number of ways of having
    # MF = MN + (M_I1 + M_I2) # NP-Hard Problem SSP (Subset Sum)
    d=0
    for MN in range(-N,N+1):
        for M_I1_D in range(-I1_D,I1_D+1,2):
            for M_I2_D in range(-I2_D,I2_D+1,2):
                if 2*MN+M_I1_D+M_I2_D == MF_D:
                    d+=1
    return d


# %%
def reachable_above_from(N,MF_D):
    sigma_plus_reachable = [(N+1,MF_D-2,i) for i in range(label_degeneracy(N+1,MF_D-2))]
    pi_reachable = [(N+1,MF_D,i) for i in range(label_degeneracy(N+1,MF_D))]
    sigma_minus_reachable = [(N+1,MF_D+2,i) for i in range(label_degeneracy(N+1,MF_D+2))]
    return (sigma_plus_reachable + pi_reachable + sigma_minus_reachable)

def twice_average_fidelity(k,g):
    return ((1 + g**2)**2 + 8*k**2*(-1 + 2*g**2) + 16*k**4)/((1 + g**2)**3 + (-8 + 20*g**2 + g**4)*k**2 + 16*k**4)

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

# def trio_transfer_efficiency(state1_label,state2_label,state3_label,bi,pulse_time=0.0001):
#     state1i = label_to_state_no(*state1_label)
#     state2i = label_to_state_no(*state2_label)
#     state3i = label_to_state_no(*state3_label)
    
#     P = state1_label[1] - state2_label[1]
#     COUPLING = COUPLINGS[P]
    
#     g = np.abs(COUPLING[bi, state1i, state3i]/COUPLING[bi, state1i, state2i])
#     k = np.abs(((ENERGIES[bi, state3i] - ENERGIES[bi, state2i]) / scipy.constants.h) / (1/pulse_time))
#     sub_transfered = twice_average_fidelity(k,g)
    
#     return sub_transfered

# def transfer_efficiency(state1_label, state2_label,bi,pulse_time=0.0001):
#     transfered = 1
    
#     state1i = label_to_state_no(*state1_label)
#     state2i = label_to_state_no(*state2_label)

#     P = (state1_label[1] - state2_label[1])*(state2_label[0] - state1_label[0])
#     this_coupling = COUPLINGS#[P]
    
#     for state3i in range(N_STATES):
#         if state3i == state1i or state3i == state2i:
#             continue
#         g = np.abs(this_coupling[bi, state1i, state3i]/this_coupling[bi, state1i, state2i])
#         k = np.abs(((ENERGIES[bi, state3i] - ENERGIES[bi, state2i]) / scipy.constants.h) / (1/pulse_time))
#         sub_transfered = twice_average_fidelity(k,g)
#         transfered *= sub_transfered
        
#     return transfered

def fidelity(ts,d=8):
    return -np.log10(1-ts+10**(-d))


# %%
INITIAL_STATE_LABELS_D = MOLECULE["StartStates_D"]
INITIAL_STATE_INDICES = [label_to_state_no(*label_d) for label_d in INITIAL_STATE_LABELS_D]
N_INITIAL_STATES = len(INITIAL_STATE_INDICES)

# %% [markdown] tags=[]
"""
# Zeeman Plot
"""

# %%
fig, axs = plt.subplots(3,1,figsize=(5,10))

N=N_MAX
for ax in axs:
    for si in range(PER_MN*(N)**2,PER_MN*(N+1)**2):
        if si in INITIAL_STATE_INDICES:
            ax.plot(B,ENERGIES[:,si],c='red',lw=0.5, alpha=0.8)
        else:
            ax.plot(B,ENERGIES[:,si],c='black',lw=0.5, alpha=0.3)
    N-=1

# %% [markdown]
"""
# Optimise 2-level
"""

# %% tags=[]
polarisation = None             # Polarisation: -1,0,1,None
initial_state_label_d = INITIAL_STATE_LABELS_D[1]   # Which state to go from
focus_state_label_d = (1,initial_state_label_d[1],0)     # Which state to highlight
desired_pulse_time = 100*1e-6   # What desired pulse time (s)
dynamic_range = 6               # What Dynamic range to use for Fidelity
#################################

if polarisation is None:
    coupling = COUPLINGS
    polarisation_text = '\pi/\sigma_\pm'
else:
    coupling = POLARISED_COUPLING[polarisation]
    polarisation_text = ['\pi','\sigma_+','\pi/\sigma_\pm','\sigma_-'][polarisation]

initial_state_index = label_to_state_no(*initial_state_label_d)
focus_state_index = label_to_state_no(*focus_state_label_d)

accessible_state_labels = reachable_above_from(initial_state_label_d[0],initial_state_label_d[1])
accessible_state_indices = [label_to_state_no(*label) for label in accessible_state_labels]
state_cmap = plt.cm.gist_rainbow(np.linspace(0,1,len(accessible_state_labels)))

fig = plt.figure(constrained_layout=True,figsize=(6,4))

gs = GridSpec(2, 3, figure=fig)
axl = fig.add_subplot(gs[0, 0])
axm = fig.add_subplot(gs[0, 1])
axr = fig.add_subplot(gs[0, 2])
axb = fig.add_subplot(gs[1, :])

fig.suptitle(r'$|{},{}\rangle_{} \rightarrow{{{}}}\rightarrow |1,M_F\rangle_i $'.format(*initial_state_label_d, polarisation_text))

axl.set_xlim(0,B_MAX/GAUSS)
axm.set_xlim(0,B_MAX/GAUSS)
axr.set_xlim(0,B_MAX/GAUSS)

axm.set_ylim(-0.1,dynamic_range+0.1)
axr.set_ylim(-0.1,dynamic_range+0.1)

axl.set_ylabel("Detuning (MHz) - 980MHz")
axm.set_ylabel("Fidelity")
axr.set_ylabel("Fidelity")
fig.supxlabel('Magnetic Field $B_z$ (G)')

# Left zeeman plot
for i, state_index in enumerate(accessible_state_indices):
    this_colour = state_cmap[i]
    det = ((ENERGIES[:, state_index] - ENERGIES[:, initial_state_index]) / scipy.constants.h)
    absg = np.abs(coupling[:, initial_state_index, state_index])
    axl.scatter(B[::10]/GAUSS, det[::10]/1e6-980, color=this_colour, edgecolors=None, alpha=absg[::10]**0.5*0.5, s=absg[::10] ** 2 * 100, zorder=2)    
    axl.plot(B/GAUSS,det/1e6-980,color='k',linewidth=0.5,zorder=3,alpha=0.3)
    
# Middle single state plot
transfered = np.ones(B_STEPS)
for off_res_index in range(N_STATES):
    if off_res_index == initial_state_index or off_res_index == focus_state_index:
        continue
    this_colour=state_cmap[accessible_state_indices.index(off_res_index)] if off_res_index in accessible_state_indices else 'black'
    for (a,b) in [(initial_state_index,focus_state_index),(focus_state_index,initial_state_index)]:
        k = np.abs((ENERGIES[:, off_res_index] - ENERGIES[:, b]) * desired_pulse_time / scipy.constants.h)
        g = np.abs(coupling[:, a, off_res_index]/coupling[:, a, b])
        sub_transfered = twice_average_fidelity(k,g)
        axm.plot(B/GAUSS,fidelity(sub_transfered, dynamic_range),c=this_colour,linestyle='dashed',linewidth=1)
        transfered *= sub_transfered
axm.plot(B/GAUSS,fidelity(transfered, dynamic_range),c=state_cmap[accessible_state_indices.index(focus_state_index)])
print(transfered[30])
    

# # Right all state plots
accessible_transfered = []
for i, focus_state_index in enumerate(accessible_state_indices):
    this_colour = state_cmap[i]
    transfered = np.ones(B_STEPS)
    for off_res_index in range(N_STATES):
        if off_res_index == initial_state_index or off_res_index == focus_state_index:
            continue
        for (a,b) in [(initial_state_index,focus_state_index),(focus_state_index,initial_state_index)]:
            k = np.abs((ENERGIES[:, off_res_index] - ENERGIES[:, b]) * desired_pulse_time / scipy.constants.h)
            g = np.abs(coupling[:, a, off_res_index]/coupling[:, a, b])
            sub_transfered = twice_average_fidelity(k,g)
            transfered *= sub_transfered
    accessible_transfered.append(transfered)
    axr.plot(B/GAUSS,fidelity(transfered, dynamic_range),c=this_colour,linewidth=1)
    

axb.set_xlim(0,B_MAX/GAUSS)
axb.set_ylim(-1,1)
# axb.set_xlabel('Magnetic Field $B_z$ (G)')
axb.set_ylabel('Magnetic Moment Difference $\Delta$ $(\mu_N)$')


axb.axhline(0, dashes=(3, 2), color='k', linewidth=1.5, alpha=1, zorder=3)
for i, focus_state_index in enumerate(accessible_state_indices):
    this_colour = state_cmap[i]
    magnetic_moment_difference = (MAGNETIC_MOMENTS[:,focus_state_index]-MAGNETIC_MOMENTS[:,initial_state_index])
    axb.plot(B/GAUSS,magnetic_moment_difference/muN, alpha=1,linewidth=1,zorder=1,c=this_colour)
    abs_magnetic_moment_difference = np.abs(magnetic_moment_difference)
    min_delta = np.argmin(abs_magnetic_moment_difference)
    if abs_magnetic_moment_difference[min_delta]/muN < 0.3:
        this_transferred = accessible_transfered[i][min_delta]
        if this_transferred < 0.5:
            continue
        text_place = B[min_delta]/GAUSS
        line_place = max(min(B[min_delta]/GAUSS,B_MAX/GAUSS*0.99),B_MAX/GAUSS*0.01)
        axb.axvline(line_place,ymin=0.5,color=this_colour,linewidth=1,dashes=(3,2))
        this_transferred = accessible_transfered[i][min_delta]
        this_transferred_string = f"{this_transferred:.4f}"
        axb.text(text_place,1.02,this_transferred_string,rotation=60,c=this_colour)

# fig.savefig('../images/2-level-optimisation.pdf')

# %% [markdown]
"""
# Magnetic moments plot
"""

# %%
fig, ax = plt.subplots(figsize=(4,6))

ax.set_xlim(0,400)
ax.set_ylim(6,0)
ax.set_xlabel('Magnetic Field $B_z$ (G)')
ax.set_ylabel('Magnetic Moment, $\mu$ $(\mu_N)$')

five_col = [
'#ff0000',
'#00ff7f',
'#00bfff',
'#0000ff',
'#ff1493'
]

MIN_MF_D = 2
MAX_MF_D = 10

states_to_plot = []
for N in range(0,2):
    Fd = 2*N + I1_D + I2_D
    for MF_D in range(max(-Fd,MIN_MF_D),min(Fd+1,MAX_MF_D+1),2):
        for di in range(label_degeneracy(N,MF_D)):
            states_to_plot.append((N,MF_D,di))

for state_label in states_to_plot:
    lw=1
    col = 'black'#five_col[int(state_label[1]/2-2)]
    ls = 'solid'
    if state_label[0] != 0:
        ls = 'dashed'
        lw=0.75
        
    index = label_to_state_no(*state_label)
    ax.plot(B/GAUSS, MAGNETIC_MOMENTS[:,index]/muN,linestyle=ls, color=col, alpha=0.65,linewidth=lw);
    

# find all triplets 
# for i,(la,lb,lc) in enumerate(ordered_states[:100]):
#     if la[0] == 1 and lb[0]==0 and lc[0]==0:
#         if la[1]<=6 and la[1]>=2 and lb[1]<=6 and lb[1]>=2 and lc[1]<=6 and lc[1]>=2:
#             fid = ordered_fidelities[i]
#             dev = ordered_deviations[i]/muN
#             if fid < 0.2 or dev > 0.1:
#                 continue
#             x = B[ordered_B[i]]/GAUSS
#             state_indices = np.array([label_to_state_no(*la),label_to_state_no(*lb),label_to_state_no(*lc)])
#             y = np.sum(np.abs(MAGNETIC_MOMENTS[ordered_B[i],state_indices]))/(3*muN)
#             ax.plot([x],[y], 'o', mfc='none',markersize=2,c='black')
#             ax.text(x+5,y,f'f={fid:.3f},d={dev:.3f}',fontsize=8,va='bottom',ha='left',picker=True)
#             ax.text(x+5,y,r'$|{},{}\rangle_{} |{},{}\rangle_{} |{},{}\rangle_{}$'.format(*la,*lb,*lc),fontsize=8,va='top',ha='left',picker=True)

# fig.savefig('../images/3-level-qubit-all-coincidences.pdf')

# %% [markdown]
"""
# Find best state Pi-pulse paths
"""

# %%
cumulative_fidelity_from_initials = np.zeros((B_STEPS,N_INITIAL_STATES,N_STATES))
predecessor_fidelity_from_initials = np.zeros((B_STEPS,N_INITIAL_STATES,N_STATES),dtype=int)

for bi in range(B_STEPS):
    considered_matrix = UNPOLARISED_PAIR_FIDELITIES[:,:,bi]
    sparse_graph = csgraph.csgraph_from_dense(-np.log(considered_matrix), null_value=np.inf) # Expecting div 0 warning, this is fine
    (distances_from_initials),(predecessors_from_initials) = csgraph.shortest_path(sparse_graph,return_predecessors=True,directed=False,indices=INITIAL_STATE_INDICES)
    
    cumulative_fidelity_from_initials[bi]=distances_from_initials
    predecessor_fidelity_from_initials[bi]=predecessors_from_initials

# %% [markdown]
"""
# Generic Optimisation Routine
"""


# %%
def maximise_fid_dev(possibilities,loop=False,plot=True,required_crossing=None,max_bi=B_STEPS,table_len=9,ignore_small_deviation=False,rate_distance=True):
    n_comb = len(possibilities)
    n_waves = len(possibilities[0]) - 1 # NOTE: assumes paths are the same length
    print(n_comb, "combinations to consider")
    possibilities_indices = np.array([np.array([label_to_state_no(*label) for label in possibility]) for possibility in possibilities])
    
    # Find best B for minimum dipole deviation
    best_deviation = np.ones((n_comb),dtype=np.double)
    best_b_index = np.ones((n_comb),dtype=int)
    for i, desired_indices in enumerate(possibilities_indices):
        all_moments = MAGNETIC_MOMENTS[:max_bi,desired_indices]
        if required_crossing is not None:
            required_deviation = all_moments[:,required_crossing[0]]-all_moments[:,required_crossing[1]]
            sign_changes = np.where(np.diff(np.sign(required_deviation)))[0]
            mask = np.ones(max_bi, dtype=bool)
            mask[sign_changes] = False
            all_moments[mask,0] = 1e10
        max_moment = np.amax(all_moments,axis=1)
        min_moment = np.amin(all_moments,axis=1)
        deviation = max_moment - min_moment
        min_diff_loc = np.argmin(deviation)
        min_diff = deviation[min_diff_loc]
        best_deviation[i] = np.abs(min_diff/muN)
        best_b_index[i] = min_diff_loc

    # Simulate microwave transfers to find fidelity *within structure*
    top_fidelities_unpol = np.zeros(n_comb,dtype=np.double)
    top_fidelities_pol = np.zeros(n_comb,dtype=np.double)
    for i, desired_indices in enumerate(possibilities_indices):
        at_Bi = best_b_index[i]
        unpol_p = 1
        pol_p = 1
        for n in range(n_waves):
            unpol_p *= UNPOLARISED_PAIR_FIDELITIES[desired_indices[n],desired_indices[n+1],at_Bi]
            pol_p *= POLARISED_PAIR_FIDELITIES[desired_indices[n],desired_indices[n+1],at_Bi]
        if loop:
            unpol_p *= UNPOLARISED_PAIR_FIDELITIES[desired_indices[0],desired_indices[-1],at_Bi]
            pol_p *= POLARISED_PAIR_FIDELITIES[desired_indices[0],desired_indices[-1],at_Bi]
        top_fidelities_unpol[i] = unpol_p
        top_fidelities_pol[i] = pol_p
        
    # Find path to get there from initial state
    top_distance = np.zeros(n_comb,dtype=np.double)
    top_distance_initial_index = np.zeros(n_comb,dtype=int)
    top_distance_topo_index = np.zeros(n_comb,dtype=int)
    for i, desired_indices in enumerate(possibilities_indices):
        at_Bi = best_b_index[i]
        
        possible_distances = cumulative_fidelity_from_initials[at_Bi,:,desired_indices]
        topo_min_index, start_min_index = np.unravel_index(possible_distances.argmin(), possible_distances.shape)
        
        top_distance[i] = np.exp(-possible_distances[topo_min_index,start_min_index])
        top_distance_initial_index[i] = start_min_index
        top_distance_topo_index[i] = topo_min_index

    # Rank state combinations
    rating = np.zeros(n_comb,dtype=np.double)
    for i in range(n_comb):
        dev = best_deviation[i]
        fid_unpol = top_fidelities_unpol[i]
        fid_pol = top_fidelities_pol[i]
        dist_unpol = top_distance[i]
        rating[i] = fidelity(fid_unpol,d=3)*fidelity(fid_pol,d=7)
        if rate_distance:
            rating[i] *= fidelity(dist_unpol,d=4)*(dist_unpol>0.95)
        if ignore_small_deviation:
            rating[i] *= dev<0.001
        else:
            rating[i] *= -np.log(dev+1e-5)
        

    order = (-rating).argsort()

    ordered_states = possibilities[order]
    ordered_B = best_b_index[order]
    ordered_fidelities_pol = top_fidelities_pol[order]
    ordered_fidelities_unpol = top_fidelities_unpol[order]
    ordered_deviations = best_deviation[order]
    ordered_distances = top_distance[order]
    ordered_distance_initial_index = top_distance_initial_index[order]
    ordered_distance_topo_index = top_distance_topo_index[order]
    ordered_rating = rating[order]
    
    # Create Table
    headers = ['States', 'Mag. Field(G)', 'MagDipDev(u)', 'UnPol-Fid', 'Pol-Fid', 'UnPol-Dist','Rating','Path']
    data = []
    for i in range(table_len):
        state_labels = ordered_states[i]
        state_numbers = np.array([label_to_state_no(*state_label) for state_label in ordered_states[i]])
        this_magnetic_field = B[ordered_B[i]]
        this_max_dev = np.abs(ordered_deviations[i])
        this_fidelity_pol = ordered_fidelities_pol[i]
        this_fidelity_unpol = ordered_fidelities_unpol[i]
        this_distance = ordered_distances[i]
        this_distance_initial_index = ordered_distance_initial_index[i]
        this_distance_topo_index = ordered_distance_topo_index[i]
        this_rating = ordered_rating[i]
        
        current_back = state_numbers[this_distance_topo_index]
        start_index = INITIAL_STATE_INDICES[this_distance_initial_index]
        initial_label = LABELS_D[current_back]
        predecessor_list = predecessor_fidelity_from_initials[ordered_B[i], this_distance_initial_index,:]
        path=f"({initial_label[0]},{initial_label[1]},{initial_label[2]})"
        path = [initial_label]
        while current_back != start_index:
            current_back = predecessor_list[current_back]
            path.append(LABELS_D[current_back])
        if len(path) < 5:
            path_string = "<".join([label_d_to_string(label) for label in path])
        else:
            path_string = "<".join([label_d_to_string(label) for label in path[:2]])
            path_string += f"<..(+{len(path)-4})..<"
            path_string += "<".join([label_d_to_string(label) for label in path[len(path)-2:]])
        states_string = ",".join([label_d_to_string(label) for label in state_labels])
        data.append([states_string,this_magnetic_field/GAUSS,this_max_dev,this_fidelity_unpol,this_fidelity_pol,this_distance,this_rating,path_string])
    print(tabulate(data, headers=headers,tablefmt="fancy_grid"))
    
    # Show magnetic moments plot
    if plot:
        fig, axs = plt.subplots(2,3,figsize=(8,8),dpi=100,sharex=True,constrained_layout=True) #,sharey=True
        i=0
        for axh in axs:
            for ax in axh:
                state_labels = ordered_states[i]
                state_numbers = np.array([label_to_state_no(*state_label) for state_label in ordered_states[i]])
                ax.set_xlim(0,B_MAX/GAUSS)
                at_mm = MAGNETIC_MOMENTS[ordered_B[i],state_numbers[0]]/muN
                ax.set_ylim(at_mm-1,at_mm+1)
                # ax.set_yticks(np.arange(-4,4,0.5))
                ax.plot(B/GAUSS,MAGNETIC_MOMENTS[:,state_numbers]/muN, alpha=1,linewidth=1.5,zorder=1);
                ax.axvline(x=min(B[ordered_B[i]]/GAUSS,B_MAX/GAUSS*0.98), dashes=(3, 2), color='k', linewidth=1.5,alpha=0.3,zorder=0)
                this_fidelity = ordered_fidelities_unpol[i]
                this_distance = ordered_distances[i]
                max_dev = np.abs(ordered_deviations[i])
                states_string = ""
                for si, label in enumerate(state_labels):
                    states_string += label_d_to_latex_string(label)
                ax.set_title(f'd={max_dev:.4f} f={this_fidelity:.4f} \n {states_string}')
                i+=1

        fig.supxlabel( 'Magnetic Field $B_z$ (G)')
        fig.supylabel('Magnetic Moment $\mu$ $(\mu_N)$')
        # fig.savefig('../images/magnetic-dipole-coincides-storage-qubit.pdf')

# %% [markdown]
"""
# Robust Storage Bit Optimisation
"""

# %%
# Find all possible combinations
# two states in N, one in N+-1

possibilities = []
for N1 in range(0,N_MAX+1): #[1]:#
    for N2 in [N1-1,N1+1]: #[0]:#
        if N2 < 0 or N2 > N_MAX:
            continue
        F1_D = 2*N1+I1_D+I2_D
        F2_D = 2*N2+I1_D+I2_D
        for MF1_D in range(-F1_D,F1_D+1,2):#[2,3,4,5]+([6] if N1>0 else []):#range(-F1,F1+1,1):
            for p1 in [-1,0,1]:
                for p2 in [-1,0,1]:
                    if MF1_D+2*p1 > F2_D or MF1_D+2*p1 < -F2_D or MF1_D+2*p2 > F2_D or MF1_D+2*p2 < -F2_D:
                        continue
                    MF2a_D = MF1_D+2*p1
                    MF2b_D = MF1_D+2*p2
                    if MF2a_D < MF2b_D:
                        continue
                    for i in range(label_degeneracy(N1,MF1_D)):
                        for j in range(label_degeneracy(N2,MF2a_D)):
                            for k in range(label_degeneracy(N2,MF2b_D)):
                                if MF2a_D == MF2b_D and j <= k:
                                    continue
                                possibilities.append([(N2,MF2a_D,j),(N1,MF1_D,i),(N2,MF2b_D,k)])
possibilities_d = np.array(possibilities)

# %%
maximise_fid_dev(possibilities_d[:,:],loop=False,required_crossing=[0,2],rate_distance=True,table_len=6)

# %% [markdown]
"""
# 4-state loop Optimisation
"""

# %% tags=[]
# Find all possible combinations
polarisations = []
for p1 in [-1,0,1]:
    for p2 in [-1,0,1]:
        for p3 in [-1,0,1]:
            for p4 in [-1,0,1]:
                if p1+p2+p3+p4 == 0:
                    polarisations.append((p1,p2,p3,p4))

state_mfs = []            
F1_D = I1_D+I2_D
for base_mf_d in range(-F1_D,F1_D+1,2):
    for p1,p2,p3,_ in polarisations:
        state_mfs.append((base_mf_d,base_mf_d+2*p1,base_mf_d+2*p1+2*p2,base_mf_d+2*p1+2*p2+2*p3))

states = []
for state_mf in state_mfs:
    for i in range(label_degeneracy(0,state_mf[0])):
        for j in range(label_degeneracy(1,state_mf[1])):
            for k in range(label_degeneracy(2,state_mf[2])):
                for l in range(label_degeneracy(1,state_mf[3])):
                    if (state_mf[1]<state_mf[3]) or (state_mf[1]==state_mf[3] and j<=l):
                        continue
                    states.append([(0,state_mf[0],i),(1,state_mf[1],j),(2,state_mf[2],k),(1,state_mf[3],l)])
                    
states=np.array(states)

# %% tags=[]
maximise_fid_dev(states,loop=True,table_len=6,rate_distance=True)

# %% [markdown]
"""
# 2-state
"""

# %% tags=[]
states=[]
for N1 in range(0,N_MAX): #[1]:#
    N2=N1+1
    F1_D = 2*N1+I1_D+I2_D
    for MF1_D in range(-F1_D,F1_D+1,2):
        for MF2_D in [MF1_D-2,MF1_D,MF1_D+2]:
            for i in range(label_degeneracy(N1,MF1_D)):
                for j in range(label_degeneracy(N2,MF2_D)):
                    states.append([(N1,MF1_D,i),(N2,MF2_D,j)])           
states=np.array(states)

# %% tags=[]
maximise_fid_dev(states,table_len=6,rate_distance=True,ignore_small_deviation=True)

# %% [markdown]
"""
# 3-state
"""

# %%
states=[]
N1=0
N2=1
N3=2
F1_D = 2*N1+I1_D+I2_D
for MF1_D in range(-F1_D,F1_D+1,2):
    for MF2_D in [MF1_D-2,MF1_D,MF1_D+2]:
        for MF3_D in [MF2_D-2,MF2_D,MF2_D+2]:
            for i in range(label_degeneracy(N1,MF1_D)):
                for j in range(label_degeneracy(N2,MF2_D)):
                    for k in range(label_degeneracy(N3,MF3_D)):
                        states.append([(N1,MF1_D,i),(N2,MF2_D,j),(N3,MF3_D,k)])           
states=np.array(states)

# %% tags=[]
maximise_fid_dev(states,table_len=9)
