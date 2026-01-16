import numpy as np
import matplotlib.pyplot as plt

dt = 0.1 # time step
omega = 1.0
T = 50

time = np.arange(0, T, dt)

x = np.zeros(time.size)
v = np.zeros(time.size)
a = np.zeros(time.size)

x[0] = 10
v[0] = 0
a[0] = -omega ** 2 * x[0]

for i in range(time.size -1):
    x[i + 1] = x[i] + v[i]*dt + 0.5 * a[i] * dt ** 2
    a[i + 1] = -omega ** 2 * x[i]
    v[i+1] = v[i] + 0.5 * (a[i] + a[i+1]) * dt

plt.plot(time, x)
plt.show()