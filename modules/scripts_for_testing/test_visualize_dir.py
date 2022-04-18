import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import numpy as np

# a = [[1,2,1,3,1,1,2,1,3], 
# 	[3,5,4,4,7,2,5,4,5], 
# 	[7,6,5,5,8,3,6,7,6],
# 	[8,7,9,7,9,8,9,9,8]]


# def df_lt(a, b, temp):
#     # if a > b
#     return torch.sigmoid((a - b) / temp)

# # sorted_err = torch.tensor(a).t().float()
# sorted_err = torch.tensor(a).float()

# thres_mu = nn.Parameter(torch.tensor([0.1]) * torch.ones(sorted_err.shape[0] - 1, 1), requires_grad=True)
# thres_sigma = nn.Parameter(torch.ones(sorted_err.shape[0] - 1, 1), requires_grad=True)

# optim = torch.optim.SGD([thres_mu, thres_sigma], lr=0.01)

# delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]

# delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)


import matplotlib.pyplot as plt
import matplotlib.tri as tri

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
AREA = 0.5 * 1 * 0.75**0.5
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

plt.figure(figsize=(8, 4))
for (i, mesh) in enumerate((triangle, trimesh)):
    plt.subplot(1, 2, i+ 1)
    plt.triplot(mesh)
    plt.axis('off')
    plt.axis('equal')

# For each corner of the triangle, the pair of other corners
pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
    return np.clip(coords, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                           np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                               for (xx, aa)in zip(x, self._alpha)])

def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):
    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.show()

draw_pdf_contours(Dirichlet([1, 1, 1]))



