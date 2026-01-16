import numpy as np
import matplotlib.pyplot as plt

L = 1.0
g = 9.81
theta = 0.2
omega = 2
dt = 0.01
c = 0.05
m = 1
T = 10

time = np.arange(0, T, dt)
angular_displacement = np.zeros(len(time))
angular_displacement[0] = theta

for i in range(len(time)-1):

    omega = omega + ((-g / L) * np.sin(angular_displacement[i]) - (c * L * omega * abs(omega))/m) * dt
    angular_displacement[i + 1] = angular_displacement[i] + omega * dt


plt.plot(time, angular_displacement)
plt.show()