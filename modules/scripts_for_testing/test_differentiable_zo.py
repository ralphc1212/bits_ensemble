import torch

a = torch.tensor([1.3308,  1.1419, -0.7699,  0.6746,  1.3610,  1.1807,  1.0082, -0.3443, 0.1054, -1.6181], requires_grad=True)

obj = torch.tensor([-1, -1, 1, 1, 1, 1, 1, 0, 1, 0]).float()

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)


optim = torch.optim.SGD([a], lr=0.01)

for ite in range(100):
	optim.zero_grad()
	grad_ = 1.
	l = 0
	b = df_lt(a, 0., 0.01)

	print('------------')
	print(b)
	for i in range(obj.shape[0]):
		if torch.round(b[i]):
			grad_ = grad_ * b[i]
			l += grad_ * (obj[i]-a[i]) ** 2
		else:
			grad_ = grad_ * (1 - b[i])

	l.backward()
	optim.step()
	print(a)