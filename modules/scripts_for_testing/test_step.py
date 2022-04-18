import numpy as np
import matplotlib.pyplot as plt

def df_ge(a, b, temp):
	return 1./(1 + np.e**((a - b)/ temp))

x = np.arange(-10, 10, 0.01)
y = df_lt(x, 5, 0.1)

plt.plot(x, y)
plt.show()