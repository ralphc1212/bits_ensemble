import torch

prob = torch.tensor([0.01])
g = torch.distributions.geometric.Geometric(prob)

for i in range(10):
	print(g.sample())

