import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
G = 6.67430e-11 # Gravitational constant in m^3 kg^-1 s^-2
m1 = 1.0 * 10 ** 24
m2 = 1.0 * 10 ** 24
m3 = 1.0 * 10 ** 22
# Initial conditions
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0 * 10 ** 8, 0.0])
v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 1.0 * 10 ** 3])
v3 = np.array([0.0, ])
d12 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
# d13 = math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
# d23 = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
# Time parameters
t_max = 3600 * 24 * 120 # 30 days
dt = 60 * 10 # 10 minutes in seconds
steps = int(t_max / dt)
# Initialize arrays to store positions
positions_m1 = np.zeros((steps, 2))
positions_m1[0] = p1
positions_m2 = np.zeros((steps, 2))
positions_m2[0] = p2
velocity_m1 = np.zeros((steps, 2))
velocity_m1[0] = v1
velocity_m2 = np.zeros((steps, 2))
velocity_m2[0] = v2
acceleration_m1 = np.zeros((steps, 2))
acceleration_m2 = np.zeros((steps, 2))

p = positions_m2[0] - positions_m1[0]
distance = np.linalg.norm(p)  # Magnitude of the distance
force_mag = G * m1 * m2 / distance ** 2  # Magnitude of the gravitational force
force_dir_12 = p / distance  # Unit vector in the direction of the force
force = force_mag * force_dir_12  # Gravitational force vector

acceleration_m1[0] = force / m1
acceleration_m2[0] = -force / m2

# Simulation loop
for i in range(steps -1):
    # update positions
    positions_m1[i+1] = positions_m1[i] + velocity_m1[i] * dt + 0.5 * acceleration_m1[i] * dt ** 2
    positions_m2[i+1] = positions_m2[i] + velocity_m2[i] * dt + 0.5 * acceleration_m2[i] * dt ** 2

    # calculate force with new positions
    p = positions_m2[i+1] - positions_m1[i+1]  # Vector from m1 to m2
    # print(positions_m2[i+1])
    distance = np.linalg.norm(p)  # Magnitude of the distance
    force_mag = G * m1 * m2 / distance ** 2  # Magnitude of the gravitational force
    force_dir_12 = p / distance  # Unit vector in the direction of the force
    force = force_mag * force_dir_12  # Gravitational force vector

    # calculate acceleration with new force
    acceleration_m1[i+1] = force / m1
    acceleration_m2[i+1] = -force / m2

    # calculate velocity with avg of old and new accelerations
    velocity_m1[i+1] = velocity_m1[i] + 0.5 * (acceleration_m1[i] + acceleration_m1[i+1]) * dt
    velocity_m2[i+1] = velocity_m2[i] + 0.5 * (acceleration_m2[i] + acceleration_m2[i+1]) * dt

plt.scatter([positions_m1[0,0]], [positions_m1[0,1]], label="M1 start")
plt.scatter([positions_m2[0,0]], [positions_m2[0,1]], label="M2 start")

M = m1 + m2
Rcm = (m1*positions_m1 + m2*positions_m2) / M

pos1_rel = positions_m1 - Rcm
pos2_rel = positions_m2 - Rcm

plt.figure(figsize=(6,6))
plt.plot(pos1_rel[:,0], pos1_rel[:,1], label="M1 (COM frame)", color='blue')
plt.plot(pos2_rel[:,0], pos2_rel[:,1], label="M2 (COM frame)", color='gray')
plt.axis("equal")
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2-Body Motion (Center-of-Mass Frame)")
plt.show()


# # Plot the orbits
# plt.figure(figsize=(6, 6))
# plt.plot(positions_m1[:, 0], positions_m1[:, 1], label="M1", color='blue')
# plt.plot(positions_m2[:, 0], positions_m2[:, 1], label="M2", color='gray')
# plt.scatter([0], [0], color='red', label="M1 (Center)")
# plt.title("M1 and M2 Orbit (2-Body Problem)")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.legend()
# plt.axis("equal")
# plt.show()