# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
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

import scipy.constants
from scipy.special import sph_harm

from sympy.physics.wigner import wigner_3j

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['text.usetex'] = True

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# %% [markdown]
"""
# Define constants
"""

# %%
# Physical Constants
HBAR = scipy.constants.hbar

# Molecular Constants
D_MOL = 1.225 * 3.33564e-30  # Molecular dipole moment (C.m)
BROT = 490.173994326310e6 * scipy.constants.h  # BROT := HBAR^2/2I

# Computational Constants
NMAX = 5  #  No. of Rotational base states 0 <= N <= NMAX

EMIN = 0
EMAX = 12  # Max E Field in kV/cm
ESTEPS = 50

# %% [markdown]
"""
# Create empty Hamiltonians
basis order  |0, 0>  |1,-1>  |1, 0>  |1,+1> ....
"""

# %%
size = 1 + 2 * NMAX + NMAX**2
# Size of matrix is no. of states, 2N+1 states for each N
Hrot = np.zeros((size, size), dtype=np.cdouble)
Hdc = np.zeros((size, size), dtype=np.cdouble)

# %% [markdown]
r"""
# Populate Rotational Hamiltonian
$$
H_{rot}=\frac{J^2}{2I}=\frac{J^2}{2\mu R^2}
$$
Which has spherical harmonic solutions
$$
\frac{J^2}{2I} |Y_{l,m}> = \frac{\hbar^2}{2I} l(l+1) |Y_{l,m}>
$$
$$
E_{l,m} = \frac{\hbar^2}{2I} l(l+1) = B_0 l(l+1)
$$
Expanding in spherical harmonic basis
$$
<Y_{l',m'}|H_{rot}|Y_{l,m}> =
\begin{cases}
    B_0 l(l+1), & \text{if}\ l'=l, m'=m \\
    0, & \text{otherwise}
\end{cases}
$$
"""

# %%
s = 0
for N in range(0, NMAX + 1):
    Erot = BROT * N * (N + 1)
    for M in range(-N, N + 1):
        Hrot[s, s] = Erot
        s += 1

# %% [markdown]
r"""
# Populate DC Stark Hamiltonian
$$H_{st} = -\bf{d}\cdot\bf{\epsilon} = -\epsilon\bf{d}\cdot\bf{\hat{z}} = -\epsilon d_{mol} \cos\theta$$
$$<Y_{l',m'}|H_{st}|Y_{l,m}> = -d_0\epsilon\int\int Y_{l',m'}^* \cos\theta Y_{l,m}$$
using the following identity where the bracketed terms are compact notation for Wigner-3j coefficients
$$\int\int Y_{A,i}(\theta,\phi)Y_{B,j}(\theta,\phi)Y_{C,k}(\theta,\phi) d^2\Omega =
\left[\frac{(2A+1)(2B+1)(2C+1)}{4\pi}\right]^{\frac{1}{2}}
\begin{pmatrix}
A & B & C\\
0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
A & B & C\\
i & j & k
\end{pmatrix}
$$
Along with the fact
$$ cos\theta = Y_{1,0}\sqrt{\frac{4\pi}{3}}$$
we get
$$<Y_{l',m'}|H_{st}|Y_{l,m}> = -d_0\epsilon\sqrt{(2l+1)(2l'+1)}(-1)^{m'}
\begin{pmatrix}
l' & 1 & l\\
-m' & 0 & m
\end{pmatrix}
\begin{pmatrix}
l' & 1 & l\\
0 & 0 & 0
\end{pmatrix}$$
This has non-zero elements only for $\Delta l = \pm 1$ and $\Delta m = 0$
"""

# %%
i = 0
j = 0
for N1 in range(0, NMAX + 1):
    for M1 in range(-N1, N1 + 1):
        for N2 in range(0, NMAX + 1):
            for M2 in range(-N2, N2 + 1):
                Hdc[i, j] = (
                    -D_MOL
                    * np.sqrt((2 * N1 + 1) * (2 * N2 + 1))
                    * (-1) ** M1
                    * wigner_3j(N1, 1, N2, -M1, 0, M2)
                    * wigner_3j(N1, 1, N2, 0, 0, 0)
                )
                j += 1
        j = 0
        i += 1

# %% [markdown]
"""
# Form and Diagonalise total Hamiltonian
"""

# %%
E = np.linspace(EMIN, EMAX, ESTEPS) * 1e5  # V/m
Htot = Hrot[..., None] + Hdc[..., None] * E
Htot = Htot.transpose(2, 0, 1)

energies, states = eigh(Htot)

# %% [markdown]
"""
# Sort states
The `eigh` function sorts the states by increasing energy and so will rearange order of vectors
"""

# %%
M = np.matrix([[1,0,0],
               [0,3,0],
               [0,0,2]])

eigval, eigvec = eigh(M)
print("eigen values:\n", eigval)
print("eigen vectors (in columns):\n", eigvec)


# %%
def sort_smooth(in_energies, in_states):
    ''' Sort states to remove false avoided crossings.

    This is a function to ensure that all eigenstates plotted change
    adiabatically, it does this by assuming that step to step the eigenstates
    should vary by only a small amount (i.e. that the  step size is fine) and
    arranging states to maximise the overlap one step to the next.

    Args:
        Energy (numpy.ndarray) : numpy.ndarray containing the eigenergies, as from numpy.linalg.eig
        States (numpy.ndarray): numpy.ndarray containing the states, in the same order as Energy
    Returns:
        Energy (numpy.ndarray) : numpy.ndarray containing the eigenergies, as from numpy.linalg.eig
        States (numpy.ndarray): numpy.ndarray containing the states, in the same order as Energy E[x,i] -> States[x,:,i]
    '''
    ls = np.arange(in_states.shape[2],dtype="int")
    number_iterations = len(in_energies[:,0])
    for i in range(1,number_iterations):
        '''
        This loop sorts the eigenstates such that they maintain some
        continuity. Each eigenstate should be chosen to maximise the overlap
        with the previous.
        '''
        #calculate the overlap of the ith and jth eigenstates
        overlaps = np.einsum('ij,ik->jk',
                                np.conjugate(in_states[i-1,:,:]),in_states[i,:,:])
        orig2 = in_states[i,:,:].copy()
        orig1 = in_energies[i,:].copy()
        #insert location of maximums into array ls
        np.argmax(np.abs(overlaps),axis=1,out=ls)
        for k in range(in_states.shape[2]):
            l = ls[k]
            if l!=k:
                in_energies[i,k] = orig1[l].copy()
                in_states[i,:,k] = orig2[:,l].copy()
    return in_energies, in_states

smooth_energies, smooth_states = sort_smooth(energies, states)

# %% [markdown]
"""
# Sort with state labels
"""

# %% [markdown]
"""
# Looking at Coefficient mixing (debug)
"""

# %%
fig=plt.figure()
state=1
coefs = []
for coefn in range(size):
    coefs.append(np.abs(states[:,coefn,state])**2)

plt.stackplot(E, coefs)

plt.xlabel("E")
plt.ylabel(r'$|c_n|^2$')
plt.ylim(0,1)
plt.show()

# %% [markdown]
"""
# Plot Energies against E strength
"""

# %%
plt.figure()
plt.xlim(0, EMAX)
#plt.ylim(-2000, 25000)
plt.ylabel("Energy/h (MHz)")
plt.xlabel("Electric Field (kV/cm)")

color = cm.rainbow(np.linspace(0, 1, NMAX+1))
c=0
n=0
for N in range(0, NMAX + 1):
    for M in range(-N, N + 1):
        plt.plot(E * 1e-5, energies[:, n] * 1e-6 / scipy.constants.h, color=color[c])
        n+=1
    c+=1

plt.show()

# %% [markdown]
"""
# Generalised Units
"""

# %% [markdown]
"""

"""

# %%
# Generalised Units code

# %% [markdown]
r"""
# Lab frame Dipole Moments

$$
<\psi_i|d_z|\psi_i> = -\frac{d}{dE} <\psi_i|H|\psi_i> = -\frac{d\epsilon_i}{dE}
$$
"""

# %%
plt.figure()
plt.xlim(0, EMAX)
plt.ylabel("Dipole Moment")
plt.xlabel("Electric Field (kV/cm)")

color = cm.rainbow(np.linspace(0, 1, NMAX+1))
c=0
n=0
for N in range(0, NMAX + 1):
    for M in range(-N, N + 1):
        plt.plot(E * 1e-5, -np.gradient(energies[:, n]), color=color[c])
        n+=1
    c+=1

plt.show()

# %% [markdown]
r"""
# Transition Dipole Moments
We can directly calculate the dipole matrix elements for $d_j=d_0 Y_{1,j}$, for $j=0,+1,-1$ representing different polarisations of light on the spherical harmonics:
$$
<N,M_N|d_j|N',M_N'> = d_0 \int\int Y_{N,M_N}^* Y_{1,j} Y_{N',M_N'} d^2\Omega
$$
Once again using the Wigner-3j coefficient symbols:
$$
<N,M_N|d_j|N',M_N'> = d_0 \sqrt{(2N+1)(2N'+1)} (-1)^{M_N}
\begin{pmatrix}
N & 1 & N'\\
-M_N & j & M_n'
\end{pmatrix}
\begin{pmatrix}
N & 1 & N'\\
0 & 0 & 0
\end{pmatrix}
$$
We can get the E field dependence by decomposing our new computed E-dependent eigenstates into spherical harmonics and summing the dipole operators for each.
$$
|N',M_N'> = \sum_{N=0}^{\infty}\sum_{M_N=-N}^{N} c_{M,M_N}Y_{M,M_N}
$$
"""

# %%
J=0

plt.figure()
plt.xlim(0, EMAX)
plt.ylim(-0.3, 0.8)
plt.ylabel("Dipole Moment ($d_0$)")
plt.xlabel("Electric Field (kV/cm)")

# Get coefficients
for state_1, state_2 in [(0,0),(2,2),(0,2),(3,3)]:
    this_dipole_moment = np.zeros(ESTEPS, dtype=np.cdouble)
    i=0
    j=0
    for N1 in range(0, NMAX + 1):
        for M1 in range(-N1, N1 + 1):
            for N2 in range(0, NMAX + 1):
                for M2 in range(-N2, N2 + 1):
                    wig = complex(wigner_3j(N1, 1, N2, -M1, J, M2) * wigner_3j(N1, 1, N2, 0, 0, 0))
                    this_dipole_moment += (
                            states[:,i,state_1] * states[:,j,state_2]
                            * np.sqrt((2 * N1 + 1) * (2 * N2 + 1))
                            * (-1) ** M1
                            * wig
                    )
                    j+=1
            j=0
            i+=1

    plt.plot(E * 1e-5, this_dipole_moment)

plt.show()

# %% [markdown]
"""
# Convergence depending on Energy level
The addition of a E-field mixes higher rotational states into lower rotational states. Because the matrix used to perform diagonalisation of our Hamiltonian is finite, it cannot have mixing of states past some N. We need to check values are convervent when leaving out these higher order rotational states
"""

# %%
converged_dipoles = []
for N in range(1, NMAX+1):
    size = 1 + 2 * N + N**2
    convergence_Htot = Hrot[:size,:size, None] + Hdc[:size,:size, None] * E
    convergence_Htot = convergence_Htot.transpose(2, 0, 1)
    convergence_energies, convergence_states = eigh(convergence_Htot)
    convergence_dipoles = -np.gradient(convergence_energies[:, 0])
    converged_dipoles.append(convergence_dipoles[-2])

fig = plt.figure()
plt.scatter(list(range(1,NMAX+1)), converged_dipoles)
plt.ylabel("Dipole Moment")
plt.xlabel("Max N in Basis")
plt.show()

# %% [markdown]
"""
# Plot Functions on a Sphere
"""


# %%
def f_sph_polar_to_cart_surf(f, resolution=50):
    # Polar and Azimuthal angles to Sample
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2*np.pi, resolution)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    # Calculate the unit sphere Cartesian coordinates of each (theta, phi).
    xyz = np.array([np.sin(theta_grid) * np.sin(phi_grid), np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid)])
    # Evaluate function over grid
    #f_grid = f(0, 1, phi_grid, theta_grid)
    f_grid = f(theta_grid, phi_grid)
    # get final output cartesian coords
    fxs, fys, fzs = np.abs(f_grid) * xyz
    return fxs, fys, fzs

def surface_plot(fxs, fys, fzs, ax):
    # Add axis lines
    ax_len = 0.5
    ax.plot([-ax_len, ax_len], [0,0], [0,0], c='0.5', lw=1)
    ax.plot([0,0], [-ax_len, ax_len], [0,0], c='0.5', lw=1)
    ax.plot([0,0], [0,0], [-ax_len, ax_len], c='0.5', lw=1)
    # Set axes limits
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    # Set camera position
    ax.view_init(elev=20, azim=45) #Reproduce view
    ax.set_xlim3d(-.35,.35)     #Reproduce magnification
    ax.set_ylim3d(-.35,.35)     #...
    ax.set_zlim3d(-.35,.35)     #...
    # Turn off Axes
    ax.axis('off')
    # Draw
    ax.plot_surface(fxs, fys, fzs, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False, alpha=0.3, shade=False)


# %%
f = lambda theta_grid, phi_grid : 0.8*sph_harm(1, 1, phi_grid, theta_grid)+0.2*sph_harm(-1, 1, phi_grid, theta_grid)
fxs, fys, fzs = f_sph_polar_to_cart_surf(f)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surface_plot(fxs, fys, fzs, ax)

# %% [markdown]
"""
# Plot the eigenstates under stark shift
"""

# %%
# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
# Create a 2-D meshgrid of (theta, phi) angles.
theta_grid, phi_grid = np.meshgrid(theta, phi)
# Calculate the unit sphere Cartesian coordinates of each (theta, phi).
xyz = np.array([np.sin(theta_grid) * np.sin(phi_grid), np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid)])

state=3
e_number=ESTEPS-1
n=0
f_grid = np.zeros((50, 50), dtype=np.cdouble)
for N in range(0, NMAX + 1):
    for M in range(-N, N + 1):
        coef = states[e_number][:,state][n]
        f_grid += coef * sph_harm(M, N, phi_grid, theta_grid)
        n+=1

# get final output cartesian coords
Yx, Yy, Yz = np.abs(f_grid) * xyz
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surface_plot(Yx, Yy, Yz, ax)

# %% [markdown]
"""
# E shifting Animation
"""

# %%
import matplotlib.gridspec as gridspec
el_max = 2

fig = plt.figure()
spec = gridspec.GridSpec(ncols=2*el_max+1, nrows=el_max+1, figure=fig, wspace=0, hspace=0)

# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
# Create a 2-D meshgrid of (theta, phi) angles.
theta_grid, phi_grid = np.meshgrid(theta, phi)
# Calculate the unit sphere Cartesian coordinates of each (theta, phi).
xyz = np.array([np.sin(theta_grid) * np.sin(phi_grid), np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid)])

#e_number=0
for e_number in [ESTEPS-1]:
    fig.clf()
    state=0
    for el in range(el_max+1):
        for m_el in range(-el, el+1):
            ax = fig.add_subplot(spec[el, m_el+el_max], projection='3d')

            n=0
            f_grid = np.zeros((50, 50), dtype=np.cdouble)
            for N in range(0, NMAX + 1):
                for M in range(-N, N + 1):
                    coef = states[e_number][:,state][n]
                    f_grid += coef * sph_harm(M, N, phi_grid, theta_grid)
                    n+=1

            Yx, Yy, Yz = np.abs(f_grid) * xyz
            surface_plot(Yx, Yy, Yz, ax)

            state+=1

    fig.suptitle(f'E = {E[e_number]*1e-5:.2f} kV/cm', fontsize=16)
    filename=f'animation/image{e_number:03}.png'
    fig.savefig(filename, dpi=300)

# %%
