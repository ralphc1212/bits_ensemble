import torch
import torch.nn as nn

nn_name = 'vgg11-8'
# params = torch.load(nn_name + '.ckpt.pytorch', map_location='cpu')
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
para_names = ['features.0.components', 'features.4.components', 'features.8.components', 'features.11.components', 'features.15.components', 'features.18.components', 'features.22.components', 'features.25.components']
channels = [3, 64, 128, 256, 256, 512, 512, 512, 512]

for i in range(len(channels) - 1):
	para_name = para_names[i]
	d_in = channels[i]
	d_out = channels[i + 1]

	output_path = './cpp/inputs/' + nn_name + '.' + para_name + '.in'
	with open(output_path, 'w') as f:
		step_size = 3
		num_member = 4
		weight_dim = d_in * d_out * 3 * 3
		f.write('\t'.join([str(step_size), str(weight_dim), str(num_member)]) + '\n')
		
		def write_2dtensor(t, write_shape=False):
			if write_shape:
				f.write('\t'.join([str(t.shape[0]), str(t.shape[1])]) + '\n')
			for i in range(t.shape[0]):
				for j in range(t.shape[1]):
					f.write(str(t[i, j].item()) + '\t')
				f.write('\n')
		
		all_bits = torch.zeros((weight_dim, num_member, step_size))
		all_code_book = torch.ones((weight_dim, step_size - 1)) * 14
		max_step = torch.ones((weight_dim, num_member)) * (step_size - 1)
		bit_count = torch.Tensor([(max_step == s).sum() * (2 ** (s + 1)) for s in range(0, step_size)])
		tot_bits = torch.sum(bit_count)
		print('# Bits for each weight:', max_step.shape)
		print('Quantized weights:', all_bits.shape)
		for s in range(step_size):
			print('Step {}, Min {}, Max {}'.format(s, torch.min(all_bits[:, :, s]).item(), torch.max(all_bits[:, :, s]).item()))
		print('Code book:', all_code_book.shape)
		print('Total Bits:', tot_bits.item(), 'bits')
		print('Full Precision Bits:', num_member * weight_dim * 32)
		print()
		write_2dtensor(max_step.int())
		write_2dtensor(all_code_book.int())
		for i in range(weight_dim):
			write_2dtensor(all_bits[i, :, :].int())

dense_para_names = ['dense1.components', 'dense2.components']
dims = [512, 256, 10]

for i in range(len(dims) - 1):
	para_name = dense_para_names[i]
	d_in = dims[i]
	d_out = dims[i + 1]

	output_path = './cpp/inputs/' + nn_name + '.' + para_name + '.in'
	with open(output_path, 'w') as f:
		step_size = 3
		num_member = 4
		weight_dim = d_in * d_out
		f.write('\t'.join([str(step_size), str(weight_dim), str(num_member)]) + '\n')
		
		def write_2dtensor(t, write_shape=False):
			if write_shape:
				f.write('\t'.join([str(t.shape[0]), str(t.shape[1])]) + '\n')
			for i in range(t.shape[0]):
				for j in range(t.shape[1]):
					f.write(str(t[i, j].item()) + '\t')
				f.write('\n')
		
		all_bits = torch.zeros((weight_dim, num_member, step_size))
		all_code_book = torch.ones((weight_dim, step_size - 1)) * 14
		max_step = torch.ones((weight_dim, num_member)) * (step_size - 1)
		bit_count = torch.Tensor([(max_step == s).sum() * (2 ** (s + 1)) for s in range(0, step_size)])
		tot_bits = torch.sum(bit_count)
		print('# Bits for each weight:', max_step.shape)
		print('Quantized weights:', all_bits.shape)
		for s in range(step_size):
			print('Step {}, Min {}, Max {}'.format(s, torch.min(all_bits[:, :, s]).item(), torch.max(all_bits[:, :, s]).item()))
		print('Code book:', all_code_book.shape)
		print('Total Bits:', tot_bits.item(), 'bits')
		print('Full Precision Bits:', num_member * weight_dim * 32)
		print()
		write_2dtensor(max_step.int())
		write_2dtensor(all_code_book.int())
		for i in range(weight_dim):
			write_2dtensor(all_bits[i, :, :].int())

