import torch

ONE = torch.tensor([1.])
ZERO = torch.tensor([0.])

phi = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
for i in range(100):
	u = torch.rand([1])
	g = torch.log(u / (1-u))
	s = torch.sigmoid((g + phi))
	print(torch.min(ONE, torch.max(ZERO, s * (1.2 + 0.2) - 0.2)))


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# import numpy as np

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # X, Y, Z = axes3d.get_test_data(0.05)
# # print(X, Y, Z)
# # print(X.shape, Y.shape, Z.shape)
# # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# # plt.show()

# x = np.expand_dims(np.arange(-1, 1, 2/10), axis=1).repeat(10, axis=1)
# y = np.transpose(x)

# z = np.minimum(1, np.maximum(0, 0.5 * (x - y) + y))
# print(x)
# print(y)
# print(z)
# ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
# ax.set_xlabel('$X$')
# ax.set_ylabel('$Y$')
# ax.set_zlabel('$Z$')
# plt.show()
