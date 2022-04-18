import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import numpy as np

# a = [1,2,5,6,9,11,15,17,18]
# a = [1,4,5,6,9,11,15,17,18]
a = [1,3,7,8,9,11,15,17,18]

# plt.figure()
# plt.hlines(1,1,20)  # Draw a horizontal line
# plt.eventplot(a, orientation='horizontal', linelengths=2, linestyles='--', colors='b')
# plt.axis('off')
# plt.show()

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

sorted_err = torch.tensor(a)

thres = nn.Parameter(torch.tensor([0.9]), requires_grad=True)

optim = torch.optim.SGD([thres], lr=0.01)

print('##### BEGIN #####')
for i in range(50):
	optim.zero_grad()
	delta_sorted_err = sorted_err[1:] - sorted_err[:-1]
	delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)

	split_point = df_lt(delta_sorted_err, torch.sigmoid(thres), 1.)
	# group_counts[idx-1] += torch.round(split_point.sum()).item()

	buf = 0.
	local_cnt = 0.
	grad_ = 1.
	t_l = []
	for iidx, err in enumerate(sorted_err):
	    buf += err
	    local_cnt += 1.
	    if iidx == len(sorted_err) - 1:
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
	print('grad:  ', thres.grad)
	optim.step()
	print('thres: ', torch.sigmoid(thres).item())
