import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def spherical_polar_plot(f, ax):
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    # Calculate the unit sphere Cartesian coordinates of each (theta, phi).
    xyz = np.array([np.sin(theta_grid) * np.sin(phi_grid), np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid)])
    # Calculate function values at points on grid
    f_grid = f(theta_grid, phi_grid)
    # get final output cartesian coords
    Yx, Yy, Yz = np.abs(f_grid) * xyz

    ax.plot_surface(Yx, Yy, Yz, rstride=2, cstride=2)