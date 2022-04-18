import torch
import torch.nn as nn

params = torch.load('ckpt.pytorch', map_location='cpu')

# get a compenents.1 out of a [compenents.0, compenents.1] pair as example.
# the size of compenents.1 is (N, int(in_channels * kernel_size * kernel_size  * k)).
# the size of compenents.0 is (N, int(out_channels * kernel_size * kernel_size  * k)).
# if first convolutional layer, k=3
# else k = int(max(out_channels, in_channels) / 8)

# convolutional channels in VGG [3, 64, 128, 256, 256, 512, 512, 512, 512]
# the first convolutional layer has in_channel 3, out_channel 64
# the second convolutional layer has in_channel 64, out_channel 128
# ......

# repeat_interleave: 
# 3 * 3 * k

# what follows is the dense layer
# dense layer dimension [512, 256, 10]
# k = max(D1, D2) * 2 / 8
# first dense layer D1: 512, D2: 256
# dense layer dimension D1: 256, D2: 10

# repeat_interleave: 
# k

v = params['features.25.components.1']
thres_means = params['features.25.thres_means.1']


v = nn.Parameter(v)
torch.nn.init.kaiming_normal_(v)

beta = torch.max(v)
alpha = torch.min(v)

def df_lt(a, b, temp):
	return torch.sigmoid((a - b) / temp)

def rbf_wmeans(x, means, sigma, eps):
	# print(x.shape)
	# print(means.shape)
	# exit()
	return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))

res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]), requires_grad=False)
# rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)

rbf_sigma = nn.Parameter(torch.tensor([0.6]), requires_grad=False)
eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

transfer_matrix = [None] * 2
split_point = [None] * 2
s = None
for idx, deno in enumerate(res_denos[:3]):
	if s is None:
		s = (beta - alpha) / deno
		vals = (s * torch.round(v / s)).unsqueeze(0)
	else:
		s = s / deno
		res_errer = v - vals.sum(0)
		print('--------'+str(idx))
		print(res_errer.mean(0)[0].mean())
		print(torch.log(res_errer.mean(0)[0].mean()/(1-res_errer.mean(0)[0].mean())))
		print(torch.median(res_errer, dim=0)[0].mean())
		print(torch.log(torch.median(res_errer, dim=0)[0].mean()/torch.median(res_errer, dim=0)[0].mean()))

		sorted_err, idx_maps = torch.sort(res_errer, dim=0)

		transfer_matrix[idx - 1] = torch.nn.functional.one_hot(idx_maps, num_classes=4).float().detach()
		delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
		delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
		# 64 is the k
		# need to be changed if for other layers
		mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(thres_means.repeat_interleave(3 * 3 * 64)), 0.01)
		split_point[idx - 1] = mean_split_point.detach()
		round_split = torch.cat([torch.zeros(1, v.shape[1]), mean_split_point])

		clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), rbf_mu, rbf_sigma, eps)
		inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+eps))
		grouped = torch.einsum('cb, cba -> ab', inner_grouped, transfer_matrix[idx - 1])
		bits = torch.round(grouped / s)
		if idx == 1:
			bits = torch.clamp(bits, min= - 2, max=2)
		if idx ==2:
			bits = torch.clamp(bits, min= - 8, max=8)
		quantized = s * bits
		
		vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

# print(transfer_matrix[0].shape)
# print(transfer_matrix[1].shape)

# print(thres_means.shape)

# print(split_point[0].shape)
# print(split_point[1].shape)
