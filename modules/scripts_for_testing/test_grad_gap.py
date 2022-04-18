import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# temperature = 0.1
# # with a fixed threshold, the value of output, grads on a2, a1, gap = a2 - a1

# a_1 = torch.tensor([1.], requires_grad=True)
# t = torch.tensor([0.5], requires_grad=True)


# a_2 = nn.Parameter(torch.arange(1.,2.,0.01), requires_grad=True)

# o = torch.sigmoid((a_2 - a_1 - t)/temperature)
# o.sum().backward()

# g_2 = a_2.grad
# g_1 = a_1.grad
# g_t = t.grad

# npa1 = a_1.detach().numpy()
# npt = t.detach().numpy()
# npa2 = a_2.detach().numpy()
# npo = o.detach().numpy()
# print(g_2)
# print(g_1)
# print(g_t)

# plt.plot(npa2, g_2)
# plt.show()

# --------

temperature = 0.1
# with a fixed threshold, the value of output, grads on a2, a1, gap = a2 - a1

# a_1 = torch.tensor([1.], requires_grad=True)
t = torch.tensor([0.5], requires_grad=True)

# a_2 = nn.Parameter(torch.arange(1., 2., 0.01), requires_grad=True)
a_1 = nn.Parameter(torch.arange(1., 2., 0.01), requires_grad=True)
a_2 = torch.tensor([2.], requires_grad=True)

o = torch.sigmoid((a_2 - a_1 - t) / temperature)
o.sum().backward()

g_2 = a_2.grad
g_1 = a_1.grad
g_t = t.grad

npa1 = a_1.detach().numpy()
npt = t.detach().numpy()
npa2 = a_2.detach().numpy()
npo = o.detach().numpy()
print(g_2)
print(g_1)
print(g_t)

plt.plot(npa1, g_1)
plt.show()