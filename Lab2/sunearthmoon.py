import numpy as np
import matplotlib.pyplot as plt
# Constants
G = 6.67430e-11 # Gravitational constant, m^3/kg/s^2
# Masses (kg)
M_sun = 1.98847e30
M_earth = 5.9722e24
M_moon = 7.342e22
# Initial positions (m)
# Sun at origin
r_sun = np.array([0.0, 0.0])
r_earth = np.array([1.496e11, 0.0]) # 1 AU
r_moon = r_earth + np.array([384.4e6, 0.0]) # Moon 384,400 km from Earth
# Initial velocities (m/s)
v_sun = np.array([0.0, 0.0])
v_earth = np.array([0.0, 29.78e3]) # Earth's orbital speed
v_moon = v_earth + np.array([0.0, 1.022e3]) # Moon's orbital speed around Earth
# Time settings
dt = 3600 # time step: 1 hour
days = 367
steps = int((days * 24 * 3600) / dt)
# Initialize arrays
r_sun_traj = np.zeros((steps, 2))
r_earth_traj = np.zeros((steps, 2))
r_moon_traj = np.zeros((steps, 2))
# Set initial positions and velocities
r_s = r_sun.copy()
r_e = r_earth.copy()
r_m = r_moon.copy()
v_s = v_sun.copy()
v_e = v_earth.copy()
v_m = v_moon.copy()
# Main integration loop (Euler method)
for i in range(steps):
    # Record positions
    r_sun_traj[i] = r_s
    r_earth_traj[i] = r_e
    r_moon_traj[i] = r_m
    # Compute distances
    r_se = np.linalg.norm(r_s - r_e)
    r_sm = np.linalg.norm(r_s - r_m)
    r_em = np.linalg.norm(r_e - r_m)
    # Compute gravitational forces
    F_se = G * M_sun * M_earth * (r_s - r_e) / r_se**3
    F_sm = G * M_sun * M_moon * (r_s - r_m) / r_sm**3
    F_em = G * M_earth * M_moon * (r_e - r_m) / r_em**3
    # Accelerations
    a_s = -(F_se + F_sm) / M_sun
    a_e = (F_se - F_em) / M_earth
    a_m = (F_sm + F_em) / M_moon
    # Update velocities and positions
    v_s += a_s * dt
    v_e += a_e * dt
    v_m += a_m * dt
    r_s += v_s * dt
    r_e += v_e * dt
    r_m += v_m * dt
# Plotting the orbits
plt.figure(figsize=(8, 8))
plt.plot(r_sun_traj[:, 0], r_sun_traj[:, 1], label="Sun")
plt.plot(r_earth_traj[:, 0], r_earth_traj[:, 1], label="Earth")
plt.plot(r_moon_traj[:, 0], r_moon_traj[:, 1], label="Moon")
plt.scatter(r_sun_traj[0, 0], r_sun_traj[0, 1], color='orange', s=100, label="Sun Start")
plt.axis("equal")
plt.title("Three-Body Problem: Sun, Earth, Moon")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()