import numpy as np
from numpy.linalg import eigh

import scipy.constants

from sympy.physics.wigner import wigner_3j

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# 1 - Define Parameters
HBAR = scipy.constants.hbar

NMAX = 4  #  No. of Rotational base states 0 <= N <= NMAX
EMAX = 12  # Max E Field in kV/cm

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
E = np.linspace(0, EMAX, int(60)) * 1e5  # V/m

Htot = Hrot[..., None] + Hdc[..., None] * E

Htot = Htot.transpose(2, 0, 1)

energies, states = eigh(Htot)

# 4 - Plot Hamiltonian
plt.figure()
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
