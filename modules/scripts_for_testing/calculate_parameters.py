import numpy as np

params = [64, 128, 256, 256, 512, 512, 512, 512]

bitensemble = []
baens = []
factor = 8

for i in range(len(params) - 1):
	latent = params[i] / factor
	bitensemble.append(4 * (params[i] * 3 * 3 * latent + params[i+1] * 3 * 3 * latent))
	baens.append(params[i] * 3 * 3 * params[i + 1] + 4 * params[i] + 4 * params[i+1])

print(np.sum(bitensemble))

print(np.sum(baens))

