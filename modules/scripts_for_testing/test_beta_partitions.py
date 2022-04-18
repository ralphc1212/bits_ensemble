import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import numpy as np

# a = [1,2,5,6,9,11,15,17,18]
# a = [1,4,5,6,9,11,15,17,18]
# a = [1,3,7,8,9,11,15,17,18]
a = [[1,3,7,8,9,11,15,17,18], 
	[1,4,5,6,9,10,11,14,15], 
	[1,2,3,4,7,9,11,12,16]]


# plt.figure()
# plt.hlines(1,1,20)  # Draw a horizontal line
# plt.eventplot(a, orientation='horizontal', linelengths=2, linestyles='--', colors='b')
# plt.axis('off')
# plt.show()

# print(np.log(0.1/0.9))
# print(torch.sigmoid(torch.tensor([-0.405])))
# print(torch.sigmoid(torch.tensor([-2.197])))

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

sorted_err = torch.tensor(a).t().float()

thres_mu = nn.Parameter(torch.tensor([0.1]) * torch.ones(sorted_err.shape[0] - 1, 1), requires_grad=True)
thres_sigma = nn.Parameter(torch.ones(sorted_err.shape[0] - 1, 1), requires_grad=True)

optim = torch.optim.SGD([thres_mu, thres_sigma], lr=0.01)

print('##### BEGIN #####')
for i in range(100):
	optim.zero_grad()

	# mean_sorted_err = sorted_err.mean(dim=1)
	delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]

	delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)

	samples = torch.distributions.normal.Normal(thres_mu, torch.sigmoid(thres_sigma)).rsample()
	splits = df_lt(delta_sorted_err, samples, 0.01)
	split_point = splits.mean(dim=1)
	# var_split_point = df_lt(torch.sigmoid(thres_var), delta_sorted_err.var(dim=1), 0.01)

	# delta_sorted_err = delta_sorted_err
	# print('----------')
	# print(delta_sorted_err)
	# print('----------')
	# mean_split_point = df_lt(delta_sorted_err.mean(dim=1), thres_mean, 0.01)
	# var_split_point = df_lt(thres_var, delta_sorted_err.var(dim=1), 0.01)

	# split_point = mean_split_point * var_split_point
	# group_counts[idx-1] += torch.round(split_point.sum()).item()

	buf = 0.
	local_cnt = 0.
	grad_ = 1.
	t_l = []
	for iidx in range(sorted_err.shape[0]):
		buf += sorted_err[iidx, :]
		local_cnt += 1.
		if iidx == sorted_err.shape[0] - 1:
			t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(int(local_cnt), -1))
		elif torch.round(split_point[iidx]):
			grad_ = grad_ * split_point[iidx]
			t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(int(local_cnt), -1))
			buf = 0.
			local_cnt = 0.
			grad_ = 1.
		else:
			grad_ = grad_ * (1 - split_point[iidx])

	# print(torch.cat(t_l, dim=0))

	loss = (sorted_err - torch.cat(t_l, dim=0))**2
	loss = loss.sum()
	loss.backward()
	print('**********')
	print('split: ', split_point)
	print('loss:  ', loss)
	print('exp_e: ', torch.cat(t_l, dim=0).detach())
	print('grad:  ', thres_mu.grad, thres_sigma.grad)
	optim.step()
	print('thres: ', torch.sigmoid(thres_mu), torch.sigmoid(thres_sigma))