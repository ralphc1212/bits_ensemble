import torch
import torch.nn as nn

batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)
def model(feature_vec):
	return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = torch.vmap(model)(examples)

