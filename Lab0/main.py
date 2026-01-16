import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 100)

y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.show()

time = np.arange(0,10,0.1)
oscillation = (2 * np.pi) * time
print(oscillation)
plt.plot(oscillation, y_cos)
plt.show()