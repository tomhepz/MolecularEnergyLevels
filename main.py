import numpy as np
from numpy.linalg import eigh
from numpy.linalg import eig

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import scipy.constants

from sympy.physics.wigner import wigner_3j

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# 1 - Define Parameters
HBAR = scipy.constants.hbar

NMAX = 7  #  No. of Rotational base states 0 <= N <= NMAX
EMAX = 12  # Max E Field in kV/cm
EMAX_STEPS = 60

D_MOL = 1.225 * 3.33564e-30  # Molecular dipole moment (C.m)
BROT = 490.173994326310e6 * scipy.constants.h  # BROT := HBAR^2/2I

# 2 - Create Hamiltonians, basis order:  |0, 0>  |1,-1>  |1, 0>  |1,+1> ....
size = 1 + 2 * NMAX + NMAX**2
# Size of matrix is no. of states, 2N+1 states for each N
Hrot = np.zeros((size, size), dtype=np.csingle)
Hdc = np.zeros((size, size), dtype=np.csingle)

# 2a - Rotational Hamiltonian
s = 0
for N in range(0, NMAX + 1):
    Erot = BROT * N * (N + 1)
    for M in range(-N, N + 1):
        Hrot[s, s] = Erot
        s += 1

# 2b - DC Stark Hamiltonian
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

# 3 - Diagonalise Hamiltonian over E
E = np.linspace(0, EMAX, int(EMAX_STEPS)) * 1e5  # V/m

Htot = Hrot[..., None] + Hdc[..., None] * E

Htot = Htot.transpose(2, 0, 1)

energies, states = eig(Htot)

# 4 - Plot Hamiltonian
fig = plt.figure()
plt.xlim(0, EMAX)
plt.ylim(-2000, 15000)
plt.ylabel("Energy/h (MHz)")
plt.xlabel("Electric Field (kV/cm)")
color = cm.rainbow(np.linspace(0, 1, NMAX+1))
c=0
n=0
for N in range(0, NMAX + 1):
    for M in range(-N, N + 1):
        plt.plot(E * 1e-5, energies[..., n] * 1e-6 / scipy.constants.h, color=color[c])
        n+=1
    c+=1
plt.show()

# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from plotting import spherical_polar_plot
plt.rc('text', usetex=True)

# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
# Create a 2-D meshgrid of (theta, phi) angles.
theta_grid, phi_grid = np.meshgrid(theta, phi)
# Calculate the unit sphere Cartesian coordinates of each (theta, phi).
xyz = np.array([np.sin(theta_grid) * np.sin(phi_grid), np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid)])
# Calculate function values at points on grid

plt.clf()

fig = plt.figure(figsize=plt.figaspect(1.))

state=3
for e_number in range(1,50):
    n=0
    f_grid = np.zeros((50, 50), dtype=np.csingle)
    for N in range(0, NMAX + 1):
        for M in range(-N, N + 1):
            coef = states[e_number][:,state][n]
            f_grid += coef * sph_harm(M, N, phi_grid, theta_grid)
            n+=1

    # get final output cartesian coords
    Yx, Yy, Yz = np.abs(f_grid) * xyz

    ax = fig.add_subplot(projection='3d')
    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1)
    # Set the Axes limits and title, turn off the Axes frame.
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

    ax.azim = 45
    ax.plot_surface(Yx, Yy, Yz, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'), linewidth=0, antialiased=False, alpha=0.3, shade=False)
    filename=f'animation/image{e_number:03}.png'
    fig.savefig(filename, dpi=100)
    fig.clf()


