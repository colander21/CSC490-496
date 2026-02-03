import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import math

from numpy import column_stack

# initial conditions
dt = 60 * 60 * 24
G = 6.67430e-11 # m^3 kg^-1 s^-2
R = 2e11
softening = 2e9

num_particles = 100_000
def init_galaxy(
        N,
        center=np.array([0.0, 0.0]),
        R_disk=4.0e10,
        R_bulge=2.0e10,
        bulge_frac=0.8,
        M_central=3.0e27,
        m_particle=1.0e10,
        G=6.67430e-11,
        M_nucleus=3.0e27,
        nucleus_velocity=np.array([0.0, 0.0])
        ):

    masses = np.full(N, m_particle)
    N_bulge = int(N * bulge_frac)
    N_disk = N - N_bulge

    bulge = np.random.normal(loc=0.0, scale=R_bulge, size=(N_bulge, 2))

    angles = np.random.rand(N_disk) * 2 * np.pi
    radii = np.sqrt(np.random.rand(N_disk)) * R_disk

    disk = column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    positions = np.vstack([bulge, disk]) + center

    relative_positions = positions - center
    r = np.linalg.norm(relative_positions, axis=1) + softening

    tangential = np.column_stack([-relative_positions[:,1], relative_positions[:,0]])
    tangential /= (np.linalg.norm(tangential, axis=1)[:, None] + 1e-30)

    vmag = np.sqrt(G * M_central / r)
    velocities = tangential * vmag[:, None]

    positions = np.vstack([positions, center.reshape(1, 2)])
    velocities = np.vstack([velocities, nucleus_velocity.reshape(1, 2)])
    masses = np.concatenate([masses, np.array([M_nucleus])])

    return positions, velocities, masses

def init_two_galaxies():
    # positions for galaxy centers
    c1 = np.array([-0.50e11, 0.0])
    c2 = np.array([0.50e11, 0.0])

    # give them opposite velocities so they approach each other
    v1 = np.array([400.0, 0.0])   # m/s
    v2 = np.array([-400.0, 0.0])

    p1, vels1, m1 = init_galaxy(
        N=num_particles,
        center=c1,
    )
    vels1 += v1

    p2, vels2, m2 = init_galaxy(
        N=num_particles,
        center=c2,
    )
    vels2 += v2

    positions = np.vstack([p1, p2])
    velocities = np.vstack([vels1, vels2])
    masses = np.concatenate([m1, m2])

    return positions, velocities, masses


positions, velocities, masses = init_two_galaxies()

def build_grid_stats(positions, velocities, masses, grid_size, R):
    grid_N = grid_size
    cell = (2*R) / grid_N

    gx = ((positions[:, 0] + R) / cell).astype(np.int64)
    gy = ((positions[:, 1] + R) / cell).astype(np.int64)

    gx = np.clip(gx, 0, grid_N-1)
    gy = np.clip(gy, 0, grid_N-1)

    M  = np.zeros((grid_N, grid_N), dtype=np.float64)
    Mx = np.zeros((grid_N, grid_N), dtype=np.float64)
    My = np.zeros((grid_N, grid_N), dtype=np.float64)
    MVx = np.zeros((grid_N, grid_N), dtype=np.float64)
    MVy = np.zeros((grid_N, grid_N), dtype=np.float64)

    # accumulate with numpy indexed add
    np.add.at(M,  (gx, gy), masses)
    np.add.at(Mx, (gx, gy), masses * positions[:, 0])
    np.add.at(My, (gx, gy), masses * positions[:, 1])
    np.add.at(MVx,(gx, gy), masses * velocities[:, 0])
    np.add.at(MVy,(gx, gy), masses * velocities[:, 1])

    COM = np.zeros((grid_N, grid_N, 2), dtype=np.float64)
    VAVG = np.zeros((grid_N, grid_N, 2), dtype=np.float64)

    mask = M > 0
    COM[mask, 0] = (Mx[mask] / M[mask])
    COM[mask, 1] = (My[mask] / M[mask])
    VAVG[mask, 0] = (MVx[mask] / M[mask])
    VAVG[mask, 1] = (MVy[mask] / M[mask])

    return M, COM, VAVG

def compute_accelerations_grid(positions, velocities, masses, R,
                               fine_G=1024, coarse_G=512,
                               Kfine=3, Kcoarse=2):
    # build grids
    Mf, COMf, Vf = build_grid_stats(positions, velocities, masses, fine_G, R)
    Mc, COMc, Vc = build_grid_stats(positions, velocities, masses, coarse_G, R)
    coarse_cells = np.argwhere(Mc > 0)  # list of (X,Y) that actually have mass

    N = positions.shape[0]
    acc = np.zeros_like(positions)

    # mapping helpers
    fine_cell = (2*R) / fine_G
    coarse_cell = (2*R) / coarse_G

    # particle cell indices in both grids
    gfx = np.clip(((positions[:,0] + R) / fine_cell).astype(np.int64), 0, fine_G-1)
    gfy = np.clip(((positions[:,1] + R) / fine_cell).astype(np.int64), 0, fine_G-1)

    gcx = np.clip(((positions[:,0] + R) / coarse_cell).astype(np.int64), 0, coarse_G-1)
    gcy = np.clip(((positions[:,1] + R) / coarse_cell).astype(np.int64), 0, coarse_G-1)

    eps2 = softening**2

    for i in range(N):
        ri = positions[i]

        # --- near field from fine grid ---
        x0, y0 = gfx[i], gfy[i]
        for x in range(max(0, x0-Kfine), min(fine_G, x0+Kfine+1)):
            for y in range(max(0, y0-Kfine), min(fine_G, y0+Kfine+1)):
                m = Mf[x, y]
                if m == 0:
                    continue
                rcell = COMf[x, y]
                dx = rcell[0] - ri[0]
                dy = rcell[1] - ri[1]
                dist2 = dx*dx + dy*dy + eps2
                inv = 1.0 / math.sqrt(dist2)
                inv3 = inv*inv*inv
                acc[i, 0] += G * m * dx * inv3
                acc[i, 1] += G * m * dy * inv3

        # --- far field from coarse grid (skip neighborhood to avoid double count) ---
        X0, Y0 = gcx[i], gcy[i]
        for X, Y in coarse_cells:
                if abs(X-X0) <= Kcoarse and abs(Y - Y0) <= Kcoarse:
                    continue
                m = Mc[X, Y]
                rcell = COMc[X, Y]
                dx = rcell[0] - ri[0]
                dy = rcell[1] - ri[1]
                dist2 = dx*dx + dy*dy + eps2
                inv = 1.0 / math.sqrt(dist2)
                inv3 = inv*inv*inv
                acc[i, 0] += G * m * dx * inv3
                acc[i, 1] += G * m * dy * inv3

    return acc


accelerations = compute_accelerations_grid(positions, velocities, masses, R)

def update_with_velocity_verlet():
    global positions, velocities, accelerations

    positions[:] = positions + velocities * dt + 0.5 * accelerations * dt**2

    new_accelerations = compute_accelerations_grid(positions, velocities, masses, R)

    velocities[:] = velocities + 0.5*(accelerations + new_accelerations) * dt

    accelerations[:] = new_accelerations

# animation
fig, ax = plt.subplots()
# limits to prevent auto scaling
R = 2e11
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_aspect("equal", adjustable="box")

scat = ax.scatter(positions[:, 0], positions[:, 1], s=1)

steps_per_frame = 2

def init():
    scat.set_offsets(positions)
    return (scat,)

def update(frame):
    for _ in range(steps_per_frame):
        update_with_velocity_verlet()
    scat.set_offsets(positions)
    return (scat,)

ani = FuncAnimation(fig, update, init_func=init, frames=850, interval=50, blit=True)

writer = FFMpegWriter(fps=30)
ani.save("Galaxy_9.mp4", writer=writer)
