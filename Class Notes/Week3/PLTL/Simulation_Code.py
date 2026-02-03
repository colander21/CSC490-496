import numpy as np
import matplotlib.pyplot as plt
def run_simulation(N=500, T=1000, dt=0.01, D=1.0, seed=None):
    """
    Simulate N non-interacting Brownian particles in 2D.
    Parameters
    ----------
    N : int
    Number of particles
    T : int
    Number of time steps
    dt : float
    Time step
    D : float
    Diffusion coefficient
    seed : int or None
    Random seed for reproducibility
    Returns
    -------
    time : ndarray, shape (T,)
    Time array
    positions : ndarray, shape (T, N, 2)
    Particle positions
    """
    if seed is not None:
        np.random.seed(seed)
    time = np.arange(T) * dt
    positions = np.zeros((T, N, 2))
    # Gaussian noise scale for diffusion
    sigma = np.sqrt(2 * D * dt)
    for t in range(1, T):
        noise = sigma * np.random.randn(N, 2)
        positions[t] = positions[t-1] + noise
    return time, positions
#Minimal Usage Example
time, positions = run_simulation(N=500, T=2000, dt=0.01, seed=0)
print("positions shape:", positions.shape)
# calculate MSE
N = 500
T = 2000
MSD_total = np.zeros(T)
for t in range(T):
    mse = (1/ N) * np.sum((positions[t, :, :] - positions[0, :, :])**2)
    MSD_total[t] = mse
print(MSD_total)
# (T, N, 2)
# Inspect Raw Data: Modify as needed
plt.figure()
# for i in range(10):
#     plt.plot(positions[:, i, 0], positions[:, i, 1])
plt.plot(MSD_total, time)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sample particle trajectories")
plt.show()