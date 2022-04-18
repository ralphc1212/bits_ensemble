import torch
import torch.nn as nn

# try first split first, as the second requires specified dropout rate.

# version 1, aggregating first

def df_lt(a, b, temp):
	# if a > b
	return torch.sigmoid((a - b) / temp)

def get_round():
    class round(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(input)
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return round().apply


#############

# N = 10
# D = 100
# import torch
# a = torch.rand((N, D)) * 10 - 5
# torch.save(a, 'test_reference.pt')
# exit()

#############


N = 10
D = 100
# reference_matrix = nn.Parameter(torch.rand((N, D)) * 10 - 5, requires_grad=True)
reference_matrix = torch.load('test_reference.pt')
res_denos = torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1])
maxBWPow = 3

thres = torch.tensor([0.8], requires_grad=True)

optim = torch.optim.SGD([thres], lr=0.5)
# thres = nn.Parameter(torch.tensor([0.8, 0.8, 0.8, 0.8]), requires_grad=True)

round = get_round()

niters = 20

with torch.autograd.set_detect_anomaly(True):

	for i in range(niters):
		optim.zero_grad()
		beta = torch.max(reference_matrix)
		alpha = torch.min(reference_matrix)
		buf = torch.zeros(D)

		group_counts = [0] * (maxBWPow - 1)

		s = None
		for idx, deno in enumerate(res_denos[:maxBWPow]):
			if s is None:
				s = (beta - alpha) / deno
				vals = (s * round(reference_matrix / s)).unsqueeze(0)
			else:
				s = s / deno
				res_errer = reference_matrix - vals.sum(0)
				aggregated_res_error = res_errer.sum(1)
				sorted_err, idx_maps = torch.sort(aggregated_res_error, dim=0)
				delta_sorted_err = sorted_err[1:] - sorted_err[:-1]
				delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
				split_point = df_lt(delta_sorted_err, thres, 0.1)
				group_counts[idx-1] += round(split_point.sum()).item()

				buf = 0.
				local_cnt = 0
				grad_ = 1.
				t_l = []
				for iidx, idx_map in enumerate(idx_maps):
					buf += res_errer[idx_map, :]
					local_cnt += 1

					if iidx == len(idx_maps) - 1:
						t_l.append((grad_ * buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
					elif round(split_point[iidx]):
						grad_ = grad_ * split_point[iidx]
						t_l.append((grad_ * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
						buf = 0.
						local_cnt = 0
					else:
						grad_ = grad_ * (1 - split_point[iidx])


				vals = torch.cat([vals, s * round(torch.cat(t_l, dim=0) / s).unsqueeze(0)], dim=0)

		loss = (reference_matrix - vals.sum(0)).mean()
		loss.backward()
		optim.step()

		print('-------------')
		print(loss.item())
		print(thres)
		print(group_counts)


#############


# (reference_matrix - vals.sum(0)).mean().backward()
# print(thres.grad)

# print((reference_matrix - vals.sum(0)).mean())

		# sorted_err, _ = torch.sort(res_errer, dim=0)
		# delta_sorted_err = sorted_err[1:, :] - sorted_err[:-1, :]
		# delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
		# split_point = df_lt(delta_sorted_err, thres[idx-1], 0.001)
		# split_voting = split_point.sum(dim=1)
		# delta_votes = split_voting[1:] - split_voting[:-1]
		# # print(split_voting.topk(3)[1])
		# print(delta_votes)
		# exit()


# import torch
# a = torch.rand((N, D)) * 10 - 5
# torch.save(a, 'test_reference.pt')


# version 2, voting



# N = 10
# D = 100
# # reference_matrix = nn.Parameter(torch.rand((N, D)) * 10 - 5, requires_grad=True)
# reference_matrix = torch.load('test_reference.pt')
# res_denos = torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1])
# maxBWPow = 3

# thres = torch.tensor([0.2], requires_grad=True)
# thres_counting = torch.tensor([0.13 * D], requires_grad=True)

# round = get_round()
# beta = torch.max(reference_matrix)
# alpha = torch.min(reference_matrix)
# buf = torch.zeros(D)

# group_counts = [0] * (maxBWPow - 1)
# s = None
# for idx, deno in enumerate(res_denos[:maxBWPow]):
# 	if s is None:
# 		s = (beta - alpha) / deno
# 		vals = (s * round(reference_matrix / s)).unsqueeze(0)
# 	else:
# 		s = s / deno
# 		res_errer = reference_matrix - vals.sum(0)
# 		sorted_err, err_idx_maps = torch.sort(res_errer, dim=0)
# 		delta_sorted_err = sorted_err[1:, :] - sorted_err[:-1, :]
# 		delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
# 		split_point = df_lt(delta_sorted_err, thres, 0.1)
# 		split_voting = split_point.sum(dim=1)
# 		total_split = df_lt(split_voting, thres_counting, 0.1)
# 		print(split_voting)
# 		group_counts[idx-1] += torch.round(total_split.sum()).item()

# 		buf = 0.
# 		local_cnt = 0
# 		t_l = []
# 		for iidx, idx_map in enumerate(err_idx_maps):
# 			buf += res_errer[idx_map, torch.arange(idx_map.shape[0])]
# 			local_cnt += 1
# 		if iidx == err_idx_maps.shape[0] - 1:
# 			t_l.append((buf / local_cnt).unsqueeze(0).expand(local_cnt, -1))
# 		elif torch.round(total_split[iidx]):
# 			t_l.append((total_split[iidx] * (buf / local_cnt)).unsqueeze(0).expand(local_cnt, -1))
# 			buf = 0.
# 			local_cnt = 0

# 		# print(t_l)
# 		vals = torch.cat([vals, s * round(torch.cat(t_l, dim=0) / s).unsqueeze(0)], dim=0)
# 		# delta_votes = split_voting[1:] - split_voting[:-1]

# print(group_counts)
# print((reference_matrix - vals.sum(0)).mean())



# #######
# def df_ge(a, b, temp):
# 	return torch.sigmoid((a - b) / temp)

# x = torch.tensor([-2., 1.], requires_grad=True)
# y = torch.nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
# opter = torch.optim.SGD([y], lr=1)

# for i in range(10):
# 	opter.zero_grad()
# 	mask = df_ge(y, x, 0.5)
# 	print('------')
# 	print(y)
# 	print(mask)

# 	loss = (mask - torch.tensor([0., 1.]))**2
# 	loss = loss.mean()
# 	loss.backward()
# 	print(y.grad)
# 	opter.step()


# import torch
# import torch.nn as nn

# def df_lt(a, b, temp):
# 	# if a > b
# 	return torch.sigmoid((a - b) / temp)

# members = torch.tensor([[1., 2., 3., 5., 6., 7., 9., 10.], [2., 3., 4., 7., 9., 10., 11., 15.]]).t()

# errs = members[1:, :] - members[:-1, :]
# thres = 1.5

# split_ = df_lt(errs, thres, 0.01)

# def calculate(difs, splits):
# 	buf = 0.
# 	cnt = 0.
# 	local_cnt = 0.
# 	t_l = []

# 	for idx, err in enumerate(difs):
# 		buf += err
# 		cnt += 1.
# 		local_cnt += 1.
# 	# 	if torch.round(splits[idx]):
# 	# 		t_l.append(buf / cnt)
# 	# 		buf = 0.
# 	# 		local_cnt = 0.

# 	t_l.append(buf / cnt)

# 	return torch.stack(t_l)

# a = torch.vmap(calculate)(errs, split_)

# print(a)




