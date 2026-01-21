import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
G = 6.67430e-11
m1 = 1.0 * 10**24
m2 = 1.0 * 10**24
m3 = 1.0 * 10**22
# Initial conditions
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0 * 10**8, 0.0])
# p3 = p2 + np.array([0.0, 5.0 * 10**6])
# p3 = np.array([-1.0 * 10**8, 0.0])
p3 = np.array([-1.0e8,  -1.0e8])
v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 1.0 * 10**3])
# v3 = np.array([math.sqrt(G * m2 / (5.0 * 10**6)), 0.0])
# v3 = np.array([-5.0 * 10**3, 0.0])
v3 = np.array([1000.0, 100.0])
d12 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
d13 = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
d23 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
# Time parameters
t_max = 3600 * 24 * 14
dt = 60
steps = int(t_max / dt)
# position arrays and initial positions
positions_m1 = np.zeros((steps, 2))
positions_m1[0] = p1
positions_m2 = np.zeros((steps, 2))
positions_m2[0] = p2
positions_m3 = np.zeros((steps, 2))
positions_m3[0] = p3

# velocity arrays and initial velocities
velocity_m1 = np.zeros((steps, 2))
velocity_m1[0] = v1
velocity_m2 = np.zeros((steps, 2))
velocity_m2[0] = v2
velocity_m3 = np.zeros((steps, 2))
velocity_m3[0] = v3

# acceleration arrays
acceleration_m1 = np.zeros((steps, 2))
acceleration_m2 = np.zeros((steps, 2))
acceleration_m3 = np.zeros((steps, 2))

# calculate forces at timestep 0
p12 = positions_m2[0] - positions_m1[0]
distance12 = np.linalg.norm(p12)  # Magnitude of the distance
force12 = G * m1 * m2 * p12 / distance12**3

p13 = positions_m3[0] - positions_m1[0]
distance13 = np.linalg.norm(p13)  # Magnitude of the distance
force13 = G * m1 * m3 * p13 / distance13**3

p23 = positions_m3[0] - positions_m2[0]
distance23 = np.linalg.norm(p23)  # Magnitude of the distance
force23 = G * m2 * m3 * p23 / distance23**3

acceleration_m1[0] = (force12 / m1) + (force13 / m1)
acceleration_m2[0] = (-force12 / m2) + (force23 / m2)
acceleration_m3[0] = (-force13 / m3) + (-force23 / m3)

# helper function to calculate kinetic energy
def kinetic_energy(masses, velocities):
    K = 0.0
    for m, v in zip(masses, velocities):
        K += 0.5 * m * np.dot(v, v)
    return K

# helper function to calculate potential energy
def potential_energy(positions, masses, G=6.67430e-11):
    U = 0.0
    N = len(masses)
    for i in range(N):
        for j in range(i+1, N):
            r = positions[j] - positions[i]
            dist = np.linalg.norm(r)
            U += -G * masses[i] * masses[j] / dist
    return U

# helper to calculate total angular momentum
def total_angular_momentum(positions, velocities, masses):
    L = 0.0
    for r, v, m in zip(positions, velocities, masses):
        L += (r[0] * (m * v[1]) - r[1] * (m * v[0]))
    return L

# helper to calculate total momentum
def total_momentum(velocities, masses):
    P = np.zeros(2)
    for v, m in zip(velocities, masses):
        P += m * v
    return P

# numpy arrays for energy tracking at each time step
masses = [m1, m2, m3]
KE = np.zeros(steps)
PE = np.zeros(steps)
E  = np.zeros(steps)
L = np.zeros(steps)
P_total = np.zeros((steps,2))

# initial values for energy graphs
pos0 = [positions_m1[0], positions_m2[0], positions_m3[0]]
vel0 = [velocity_m1[0], velocity_m2[0], velocity_m3[0]]
KE[0] = kinetic_energy(masses, vel0)
PE[0] = potential_energy(pos0, masses, G=G)
E[0]  = KE[0] + PE[0]
L[0] = total_angular_momentum(pos0, vel0, masses)
P_total[0] = total_momentum(vel0, masses)

# Simulation loop
for i in range(steps -1):
    # update positions
    positions_m1[i+1] = positions_m1[i] + velocity_m1[i] * dt + 0.5 * acceleration_m1[i] * dt ** 2
    positions_m2[i+1] = positions_m2[i] + velocity_m2[i] * dt + 0.5 * acceleration_m2[i] * dt ** 2
    positions_m3[i + 1] = positions_m3[i] + velocity_m3[i] * dt + 0.5 * acceleration_m3[i] * dt ** 2

    # calculate forces with new positions
    p12 = positions_m2[i+1] - positions_m1[i+1]
    distance12 = np.linalg.norm(p12)  # Magnitude of the distance
    force12 = G * m1 * m2 * p12 / distance12**3

    p13 = positions_m3[i+1] - positions_m1[i+1]
    distance13 = np.linalg.norm(p13)  # Magnitude of the distance
    force13 = G * m1 * m3 * p13 / distance13**3

    p23 = positions_m3[i+1] - positions_m2[i+1]
    distance23 = np.linalg.norm(p23)  # Magnitude of the distance
    force23 = G * m2 * m3 * p23 / distance23**3

    # calculate acceleration with new force
    acceleration_m1[i+1] = (force12 / m1) + (force13 / m1)
    acceleration_m2[i+1] = (-force12 / m2) + (force23 / m2)
    acceleration_m3[i+1] = (-force13 / m3) + (-force23 / m3)

    # calculate velocity with avg of old and new accelerations
    velocity_m1[i+1] = velocity_m1[i] + 0.5 * (acceleration_m1[i] + acceleration_m1[i+1]) * dt
    velocity_m2[i+1] = velocity_m2[i] + 0.5 * (acceleration_m2[i] + acceleration_m2[i+1]) * dt
    velocity_m3[i+1] = velocity_m3[i] + 0.5 * (acceleration_m3[i] + acceleration_m3[i+1]) * dt

    # calculate energy values at new positions and velocities
    pos = [positions_m1[i + 1], positions_m2[i + 1], positions_m3[i + 1]]
    vel = [velocity_m1[i + 1], velocity_m2[i + 1], velocity_m3[i + 1]]
    KE[i + 1] = kinetic_energy(masses, vel)
    PE[i + 1] = potential_energy(pos, masses, G=G)
    E[i + 1] = KE[i + 1] + PE[i + 1]
    L[i + 1] = total_angular_momentum(pos, vel, masses)
    P_total[i + 1] = total_momentum(vel, masses)


M = m1 + m2 + m3
Rcm = (m1*positions_m1 + m2*positions_m2 + m3*positions_m3) / M

pos1_rel = positions_m1 - Rcm
pos2_rel = positions_m2 - Rcm
pos3_rel = positions_m3 - Rcm

plt.figure(figsize=(6,6))
plt.plot(pos1_rel[:,0], pos1_rel[:,1], label="M1 (COM frame)", color='blue')
plt.plot(pos2_rel[:,0], pos2_rel[:,1], label="M2 (COM frame)", color='gray')
plt.plot(pos3_rel[:,0], pos3_rel[:,1], label="M3 (COM frame)", color='orange')
plt.axis("equal")
plt.legend()
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("3-Body Motion (Center-of-Mass Frame)")
plt.show()



plt.figure()
plt.plot((E - E[0]) / abs(E[0]))
plt.xlabel("step"); plt.ylabel("Relative energy error"); plt.title("Energy conservation")
plt.show()

plt.figure()
plt.plot(L - L[0] / abs(L[0]))
plt.xlabel("step"); plt.ylabel("Angular momentum change"); plt.title("Angular momentum")
plt.show()

plt.figure()
plt.plot(P_total[:,0], label="Px"); plt.plot(P_total[:,1], label="Py")
plt.legend(); plt.title("Total linear momentum (should be constant)")
plt.show()
