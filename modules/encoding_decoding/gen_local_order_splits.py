import torch
import torch.nn as nn

nn_name = 'vgg11'
params = torch.load(nn_name + '.ckpt.pytorch', map_location='cpu')
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

for para_name, _ in params.items():
	if 'components' not in para_name:
		continue
	para_list = para_name.split('.')
	if 'features' in para_name:
		if para_list[1] == '0':
			k = 3 * 3 * 3
		elif para_list[1] == '4':
			k = 3 * 3 * 16
		elif para_list[1] == '8' or para_list[1] == '11':
			k = 3 * 3 * 32
		else:
			k = 3 * 3 * 64
	else:
		k = 128 if para_list[0] == 'dense1' else 64

	v = params[para_name]
	thres_means = params[para_name.replace('components', 'thres_means')][int(para_list[-1])]

	v = nn.Parameter(v)
	torch.nn.init.kaiming_normal_(v)

	beta = torch.max(v)
	alpha = torch.min(v)

	def df_lt(a, b, temp):
			# if a > b
			return torch.sigmoid((a - b) / temp)

	def rbf_wmeans(x, means, sigma, eps):
			return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))

	def get_code_book(clustering_vec):
		clustering_vec = torch.round(clustering_vec).permute(1, 0, 2)
		sum_clustering_vec = clustering_vec.sum(1).unsqueeze(-1)
		cum_mat = torch.ones_like(clustering_vec).triu().transpose(1, 2)
		cum_sum_clustering_vec = cum_mat.matmul(sum_clustering_vec)
		code_vec = clustering_vec.matmul(cum_sum_clustering_vec).squeeze()
		if code_vec.shape[1] == 4:
			code_book = []
			for i in range(code_vec.shape[0]):
				vec = code_vec[i]
				if vec[0] == vec[1] and vec[0] == vec[2] and vec[0] == vec[3]:
					code_book.append(0)
				elif vec[0] == vec[1] and vec[1] == vec[2]:
					code_book.append(1)
				elif vec[0] == vec[1] and vec[1] == vec[3]:
					code_book.append(2)
				elif vec[0] == vec[2] and vec[2] == vec[3]:
					code_book.append(3)
				elif vec[1] == vec[2] and vec[2] == vec[3]:
					code_book.append(4)
				elif vec[0] == vec[1] and vec[2] == vec[3]:
					code_book.append(5)
				elif vec[0] == vec[2] and vec[1] == vec[3]:
					code_book.append(6)
				elif vec[0] == vec[3] and vec[1] == vec[2]:
					code_book.append(7)
				elif vec[0] == vec[1]:
					code_book.append(8)
				elif vec[0] == vec[2]:
					code_book.append(9)
				elif vec[0] == vec[3]:
					code_book.append(10)
				elif vec[1] == vec[2]:
					code_book.append(11)
				elif vec[1] == vec[3]:
					code_book.append(12)
				elif vec[2] == vec[3]:
					code_book.append(13)
				else:
					code_book.append(14)
			return torch.Tensor(code_book).unsqueeze(0)
		else:
			print('Not Implemented')
			exit()
			return None

	res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

	rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
	rbf_sigma = nn.Parameter(torch.tensor([0.6]), requires_grad=False)
	eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

	transfer_matrix = [None] * 2
	split_point = [None] * 2
	s = None
	output_path = './cpp/inputs/' + nn_name + '.' + para_name + '.in'
	with open(output_path, 'w') as f:
		step_size = 3
		num_member = v.shape[0]
		weight_dim = v.shape[1]
		f.write('\t'.join([str(step_size), str(weight_dim), str(num_member)]) + '\n')		
		def write_2dtensor(t, write_shape=False):
			if write_shape:
				f.write('\t'.join([str(t.shape[0]), str(t.shape[1])]) + '\n')
			for i in range(t.shape[0]):
				for j in range(t.shape[1]):
					f.write(str(t[i, j].item()) + '\t')
				f.write('\n')
		
		all_code_book = None
		all_bits = None
		for idx, deno in enumerate(res_denos[:step_size]):
			if s is None:
				s = (beta - alpha) / deno
				vals = (s * torch.round(v / s)).unsqueeze(0)
				all_bits = torch.round(v / s).unsqueeze(0)
			else:
				s = s / deno
				res_errer = v - vals.sum(0)

				sorted_err, idx_maps = torch.sort(res_errer, dim=0)

				transfer_matrix[idx - 1] = torch.nn.functional.one_hot(idx_maps, num_classes=4).float().detach()
				delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
				delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
				mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(thres_means.repeat_interleave(k)), 0.01)
				split_point[idx - 1] = mean_split_point.detach()
				round_split = torch.round(torch.cat([torch.zeros(1, v.shape[1]), mean_split_point]))

				clustering_vec = torch.round(rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), rbf_mu, rbf_sigma, eps))
				inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+eps))
				grouped = torch.einsum('cb, cba -> ab', inner_grouped, transfer_matrix[idx - 1])
				bits = torch.round(grouped / s)
				if idx == 1:
					bits = torch.clamp(bits, min=-2, max=2)
				if idx == 2:
					bits = torch.clamp(bits, min=-8, max=8)
				quantized = s * bits
				vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)
				all_bits = torch.cat([all_bits, bits.unsqueeze(0)], dim=0)
				code_book = get_code_book(clustering_vec)
				all_code_book = torch.cat([all_code_book, code_book], dim=0) if all_code_book != None else code_book
		all_bits = all_bits.permute(2, 1, 0)
		all_code_book = all_code_book.t()
		max_step = torch.zeros((weight_dim, num_member))
		for i in range(weight_dim):
			for j in range(num_member):
					for s in range(step_size - 1, 1, -1):
						if all_bits[i, j, s] != 0:
							max_step[i, j] = s
							break
		bit_count = torch.Tensor([(max_step == s).sum() * (2 ** (s + 1)) for s in range(0, step_size)])
		tot_bits = torch.sum(bit_count)
		print('# Bits for each weight:', max_step.shape)
		print('Quantized weights:', all_bits.shape)
		for s in range(step_size):
			print('Step {}, Min {}, Max {}'.format(s, torch.min(all_bits[:, :, s]).item(), torch.max(all_bits[:, :, s]).item()))
		print('Code book:', all_code_book.shape)
		print('Total Bits:', tot_bits.item(), 'bits')
		write_2dtensor(max_step.int())
		write_2dtensor(all_code_book.int())
		for i in range(weight_dim):
			write_2dtensor(all_bits[i, :, :].int())
