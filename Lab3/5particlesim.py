import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
# Gravitational constant
G = 6.67430e-11 # m^3 kg^-1 s^-2
# Number of particles
num_particles = 5
# Masses of the particles (kg)
masses = np.array([1e10, 2e10, 1.5e10, 3e10, 2.5e10])
# Initial positions (in meters)
positions = np.random.rand(num_particles, 2) * 1e11 # Random positions within a 100 billion meter range
# Initial velocities (in meters per second)
velocities = np.random.randn(num_particles, 2) * 1e3 # Random velocities
# Time step for simulation (seconds)
dt = 60 * 60 * 24 # 1 day
#dt = 60 * 60 * 24 * 5 # 5 days
# Function to calculate gravitational force between two particles
def gravitational_force(m1, m2, r1, r2):
    r = r2 - r1
    distance = np.linalg.norm(r)
    force_magnitude = G * m1 * m2 / distance ** 2
    force = force_magnitude * r / distance # Direction of the force
    return force
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.2e11, 1.2e11)
ax.set_ylim(-1.2e11, 1.2e11)
# Create plot objects for particles
scat = ax.scatter(positions[:, 0], positions[:, 1], s=50, c='blue')
# Function to update the positions for the animation
def update(frame):
    global positions, velocities
    # Create an array to store the net forces on each particle
    forces = np.zeros_like(positions)
    # Calculate forces between each pair of particles
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            force = gravitational_force(masses[i], masses[j], positions[i], positions[j])
            forces[i] += force
            forces[j] -= force # Newton's third law (action and reaction)
        # Update velocities based on the forces (F = ma, so a = F/m)
        accelerations = forces / masses[:, None] # Divide force by mass for each particle
        velocities += accelerations * dt
        # Update positions based on velocities
        positions += velocities * dt
        # Update the scatter plot data
        scat.set_offsets(positions)
        return scat,
# Create the animation
ani = FuncAnimation(fig, update, frames=365, interval=50, blit=True)
# Save the animation as an mp4 video using FFMpegWriter
writer = FFMpegWriter(fps=30)
ani.save("FiveParticles.mp4", writer=writer)
plt.show()