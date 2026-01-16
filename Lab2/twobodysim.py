import numpy as np
import matplotlib.pyplot as plt
# Constants
G = 6.67430e-11 # Gravitational constant in m^3 kg^-1 s^-2
m1 = 5.972e24 # Mass of the Earth (kg)
m2 = 7.348e22 # Mass of the Moon (kg)
d = 3.844e8 # Initial distance between Earth and Moon (m)
# Initial conditions
x1 = np.array([0.0, 0]) # Position of Earth (fixed at origin)
x2 = np.array([d, 0]) # Position of Moon
v1 = np.array([0.0, 0]) # Velocity of Earth (fixed at rest)
v2 = np.array([0, 1.022e3]) # Velocity of Moon (in the positive y-direction)
# Time parameters
t_max = 3600 * 24 * 120 # 30 days
dt = 60 * 10 # 10 minutes in seconds
steps = int(t_max / dt)
# Initialize arrays to store positions
positions_earth = np.zeros((steps, 2))
positions_moon = np.zeros((steps, 2))
# Simulation loop
for i in range(steps):
# Compute the gravitational force between Earth and Moon
    x = x2 - x1 # Vector from Earth to Moon
    distance = np.linalg.norm(x) # Magnitude of the distance
    force_mag = G * m1 * m2 / distance**2 # Magnitude of the gravitational force
    force_dir = x / distance # Unit vector in the direction of the force
    force = force_mag * force_dir # Gravitational force vector
    # Update positions based on the velocities
    x2 += v2 * dt # Update Moon's position
    x1 += v1 * dt # Update Earth's position
    # Update velocities based on the force (Newton's Second Law)
    v2 -= force / m2 * dt # Update Moon's velocity (force = m2 * acceleration)
    #v1 += force / m1 * dt # Update Earth's velocity (force = m1 * acceleration)
    # Save positions for plotting
    positions_earth[i] = x1
    positions_moon[i] = x2
# Plot the orbits
plt.figure(figsize=(6, 6))
plt.plot(positions_earth[:, 0], positions_earth[:, 1], label="Earth", color='blue')
plt.plot(positions_moon[:, 0], positions_moon[:, 1], label="Moon", color='gray')
plt.scatter([0], [0], color='red', label="Earth (Center)")
plt.title("Earth and Moon Orbit (2-Body Problem)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.show()
