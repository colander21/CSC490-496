import numpy as np
import matplotlib.pyplot as plt

#ball dropping (euler-cromer method)
# a = delta v / delta t
# (vi+1 - vi) / (delta t) = g

# vi+1 = vi + g * delta t
# hi+1 = hi - vi * delta t

dt = 2 # time step
g = 9.81 # Gravity
h = 100 # initial height
v = 0 # initial velocity

time = np.arange(0, 5, dt)
height = np.zeros(len(time))
height[0] = 100

for i in range(len(time) -1):
    v = v + g * dt
    h = height[i] - v * dt
    if height[i+1] < 0:
        break
    height[i+1] = h

plt.plot(time, height)
plt.show()