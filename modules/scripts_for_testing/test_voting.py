import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import numpy as np

# a = [1,2,5,6,9,11,15,17,18]
# a = [1,4,5,6,9,11,15,17,18]
a = [[1,3,7,8,9,11,15,17,18], 
	[1,4,5,6,9,10,11,14,15], 
	[1,2,3,4,7,9,11,12,16]]

# plt.figure()
# plt.hlines(1,1,20)  # Draw a horizontal line
# plt.eventplot(a, orientation='horizontal', linelengths=2, linestyles='--', colors='b')
# plt.axis('off')
# plt.show()

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

sorted_err = torch.tensor(a)

thres = nn.Parameter(torch.tensor([[0.5], [0.5], [0.5]]), requires_grad=True)
th2 = nn.Parameter(torch.tensor([2.]), requires_grad=True)

optim = torch.optim.SGD([thres, th2], lr=0.01)

print('##### BEGIN #####')
for i in range(20):
	optim.zero_grad()
	delta_sorted_err = sorted_err[:,1:] - sorted_err[:,:-1]
	print('**********')
	delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
	split_point = df_lt(delta_sorted_err, torch.sigmoid(thres), 0.1)

	# group_counts[idx-1] += torch.round(split_point.sum()).item()
	split_vote = split_point.sum(0)

	split_vote = df_lt(split_vote, th2, 1.)
	# print(split_vote)

	buf = 0.
	local_cnt = 0.
	grad_ = 1.
	t_l = []
	# print(sorted_err.shape)
	for iidx in range(sorted_err.shape[1]):
		buf += sorted_err[:, iidx]
		local_cnt += 1.
		if iidx == sorted_err.shape[1] - 1:
			t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(int(local_cnt), -1))
		elif torch.round(split_vote[iidx]):
			grad_ = grad_ * split_vote[iidx]
			t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(int(local_cnt), -1))
			buf = 0.
			local_cnt = 0.
			grad_ = 1.
		else:
			grad_ = grad_ * (1 - split_vote[iidx])

	loss = (sorted_err - torch.cat(t_l, dim=0).t())**2
	loss = loss.sum()
	loss.backward()

	print('vote: ', split_vote)
	print('split: ', split_vote > th2)
	print('loss:  ', loss)
	print('exp_e: ', torch.cat(t_l, dim=0).detach())
	print('grad:  ', thres.grad)
	optim.step()
	print('thres: ', torch.sigmoid(thres))
	print('thres2: ', torch.sigmoid(th2))
