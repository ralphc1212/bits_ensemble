import torch
import torch.nn as nn

# only weight quantization, without decomposition

TEMPERATURE = 0.01
THRES = 0.9
LTIMES = 8

# batch ensemble - 如何把ensemble放进一个神经网络里/如何one pass得到ensemble的prediction
# bayesian bits - 一种优美的量化方式，google18 integer-only-inference-quantization
# up or down - 一种post training quantization的方法，决定round up or down，我们决定分组

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

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

def rbf_wmeans(x, means, sigma, eps):
    # return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))
    return torch.exp(- (x - means + eps).pow(2).pow(0.5) / (2 * sigma ** 2))

class dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True, bias=False):
        super(dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.use_bias = bias

        k = int(max(self.D1, self.D2) * 2 / LTIMES)
        self.k = k

        # print('comparisions: {}. {}.'.format(int(N * D1 * k) + int(N * D2 * k), 
        #     int(N * D1 * D2)))

        self.K = nn.Parameter(torch.normal(0, 1, (N, int(D1 * k))), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (N, int(D2 * k))), requires_grad=True)

        torch.nn.init.kaiming_normal_(self.K)
        torch.nn.init.kaiming_normal_(self.V)

        self.components = nn.ParameterList([self.K, self.V])

        # self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.ParameterList([nn.Parameter(torch.zeros(1, self.K.shape[1]), requires_grad=False), 
            nn.Parameter(torch.zeros(1, self.V.shape[1]), requires_grad=False)])

        self.thres_means = nn.ParameterList([
            nn.Parameter(torch.tensor([[-1.3] * D1] * (self.maxBWPow - 1)), requires_grad=True) , 
            nn.Parameter(torch.tensor([[-1.3] * D2] * (self.maxBWPow - 1)), requires_grad=True)
        ])

        # self.thres_means = nn.ParameterList([
        #     nn.Parameter(torch.tensor([[-4.] * D1, [-3.] * D1]), requires_grad=True) , 
        #     nn.Parameter(torch.tensor([[-4.] * D2, [-3.] * D2]), requires_grad=True)
        # ])


        self.update_partition = True
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.normal(0, 1, (N, D2)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.bias)
            self.components.append(self.bias)
            self.zeroD.append(nn.Parameter(torch.zeros(1, self.bias.shape[1]), requires_grad=False))
            self.thres_means.append(nn.Parameter(torch.tensor([[-1.3] * D2] * (self.maxBWPow - 1)), requires_grad=True))
            self.transfer_matrix = [None] * ((self.maxBWPow - 1) * 3)
            self.split_point = [None] * ((self.maxBWPow - 1) * 3)
        else:
            self.transfer_matrix = [None] * ((self.maxBWPow - 1) * 2)
            self.split_point = [None] * ((self.maxBWPow - 1) * 2)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.6]), requires_grad=False)
        self.clamp_value = [2, 8]

        self.temperature = TEMPERATURE

    def quant(self, weight_idx):
        beta = torch.max(self.components[weight_idx])
        alpha = torch.min(self.components[weight_idx])

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.components[weight_idx] / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.components[weight_idx] - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix[int(weight_idx * 2) + idx - 1] = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    if self.use_bias and weight_idx == 2:
                        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_means[weight_idx][idx - 1]), 0.01)
                        # self.split_point[int(weight_idx * 3) + idx - 1]  = mean_split_point.detach()
                    else:
                        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_means[weight_idx][idx - 1].repeat_interleave(self.k)), 0.01)
                    self.split_point[int(weight_idx * 2) + idx - 1]  = mean_split_point.detach()

                    self.npartitions[weight_idx][idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(mean_split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD[weight_idx], mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix[int(weight_idx * 2) + idx - 1])

                    self.npartitions[weight_idx][idx - 1] += torch.round(self.split_point[int(weight_idx * 2) + idx - 1]).mean(dim=1).sum().item()
                    round_split = torch.cat([self.zeroD[weight_idx], self.split_point[int(weight_idx * 2) + idx - 1]])

                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), self.rbf_mu, self.rbf_sigma, self.eps)
                clustering_vec = self.round(clustering_vec)

                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+self.eps))
                grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix[int(weight_idx * 2) + idx - 1])

                bits = self.round(grouped / s)

                bits = torch.clamp(bits, min= - self.clamp_value[idx - 1], max=self.clamp_value[idx - 1])
                quantized = s * bits

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        return vals.sum(0)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        if self.use_bias:
            self.npartitions = [[0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1)]
        else:
            self.npartitions = [[0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1)]

        quant_K = self.quant(0).view(self.N, self.D1, self.k)
        quant_V = self.quant(1).view(self.N, self.D2, self.k)
        w = torch.einsum('nik, nok -> nio', quant_K, quant_V)
        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if self.use_bias:
            quant_bias = self.quant(2).unsqueeze(0)

            act += quant_bias

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

class Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_kv_gn_baens, self).__init__()

        self.first = first
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        D = int(in_channels * out_channels * kernel_size * kernel_size)
        self.D = D
        self.use_bias = bias

        if first:
            k = 3
        else:
            k = int(max(self.out_channels, self.in_channels) / LTIMES)
        self.k = k

        self.K = nn.Parameter(torch.normal(0, 1, (N, int(out_channels * kernel_size * kernel_size * k))), requires_grad=True)
        self.V = nn.Parameter(torch.normal(0, 1, (N, int(in_channels * kernel_size * kernel_size  * k))), requires_grad=True)

        # print('comparisions: {}. {}.'.format(int(N * out_channels * kernel_size * kernel_size * k) + int(N * in_channels * kernel_size * kernel_size * k), 
        #     int(N * out_channels * in_channels * kernel_size * kernel_size)))

        # print(min(self.out_channels, self.in_channels))
        # print(self.k)
        # print(self.K.shape)
        # print(self.V.shape)
        torch.nn.init.kaiming_normal_(self.K)
        torch.nn.init.kaiming_normal_(self.V)

        self.components = nn.ParameterList([self.K, self.V])

        # self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        # self.quant = res_quant(maxBWPow=3)
        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.ParameterList([nn.Parameter(torch.zeros(1, self.K.shape[1]), requires_grad=False), 
            nn.Parameter(torch.zeros(1, self.V.shape[1]), requires_grad=False)])

        self.thres_means = nn.ParameterList([
            nn.Parameter(torch.tensor([[-1.3] * out_channels] * (self.maxBWPow - 1)), requires_grad=True) , 
            nn.Parameter(torch.tensor([[-1.3] * in_channels] * (self.maxBWPow - 1)), requires_grad=True)
        ])

        # self.thres_means = nn.ParameterList([
        #     nn.Parameter(torch.tensor([[-4.] * out_channels, [-3.] * out_channels]), requires_grad=True) , 
        #     nn.Parameter(torch.tensor([[-4.] * in_channels, [-3.] * in_channels]), requires_grad=True)
        # ])

        # self.thres_mean = nn.Parameter(torch.tensor([-1.3] * out_channels), requires_grad=True)
        # self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)

        self.update_partition = True
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.normal(0, 1, (N, out_channels)), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.bias)
            self.components.append(self.bias)
            self.zeroD.append(nn.Parameter(torch.zeros(1, self.bias.shape[1]), requires_grad=False))
            self.thres_means.append(nn.Parameter(torch.tensor([[-1.3] * out_channels] * (self.maxBWPow - 1)), requires_grad=True))
            self.transfer_matrix = [None] * ((self.maxBWPow - 1) * 3)
            self.split_point = [None] * ((self.maxBWPow - 1) * 3)
        else:
            self.transfer_matrix = [None] * ((self.maxBWPow - 1) * 2)
            self.split_point = [None] * ((self.maxBWPow - 1) * 2)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.6]), requires_grad=False)
        self.clamp_value = [2, 8]

        self.temperature = TEMPERATURE

    def quant(self, weight_idx):
        beta = torch.max(self.components[weight_idx])
        alpha = torch.min(self.components[weight_idx])

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.components[weight_idx] / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.components[weight_idx] - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)

                    self.transfer_matrix[int(weight_idx * 2) + idx - 1] = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)

                    if self.use_bias and weight_idx == 2:
                        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_means[weight_idx][idx - 1]), 0.01)
                        # self.split_point[int(weight_idx * 3) + idx - 1]  = mean_split_point.detach()
                    else:
                        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_means[weight_idx][idx - 1].repeat_interleave(self.kernel_size * self.kernel_size * self.k)), 0.01)

                    self.split_point[int(weight_idx * 2) + idx - 1]  = mean_split_point.detach()

                    self.npartitions[weight_idx][idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(mean_split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD[weight_idx], mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix[int(weight_idx * 2) + idx - 1])

                    self.npartitions[weight_idx][idx - 1] += torch.round(self.split_point[int(weight_idx * 2) + idx - 1]).mean(dim=1).sum().item()
                    
                    round_split = torch.cat([self.zeroD[weight_idx], self.split_point[int(weight_idx * 2) + idx - 1]])

                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), self.rbf_mu, self.rbf_sigma, self.eps)
                clustering_vec = self.round(clustering_vec)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+self.eps))

                grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix[int(weight_idx * 2) + idx - 1])

                bits = self.round(grouped / s)
                bits = torch.clamp(bits, min= - self.clamp_value[idx - 1], max=self.clamp_value[idx - 1])

                quantized = s * bits

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        return vals.sum(0)

    def forward(self, x):
        # if not self.first:
        #     x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        # self.npartitions = [0] * (self.maxBWPow - 1)
        if self.use_bias:
            self.npartitions = [[0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1)]
        else:
            self.npartitions = [[0] * (self.maxBWPow - 1), [0] * (self.maxBWPow - 1)]


        quant_K = self.quant(0).view(self.N, self.out_channels, self.kernel_size, self.kernel_size, self.k)
        quant_V = self.quant(1).view(self.N, self.in_channels, self.kernel_size, self.kernel_size, self.k)

        w = torch.einsum('nopqk, nipqk -> noipq', quant_K, quant_V)

        # print(w.shape)
        # print(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)
        w = w.reshape(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

        # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
        act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)
        # act = act.reshape(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])

        if self.use_bias:
            quant_bias = self.quant(2).view(1, -1, 1, 1)
            act += quant_bias

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

# import torch
# from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
# values = torch.tensor([[5.,2.,3.,1.,4.,5.]], requires_grad=True)
# optim = torch.optim.SGD([values], lr=0.1)
# for i in range(100):
#     optim.zero_grad()
#     out = soft_sort(values, regularization_strength=0.1)
#     l = (out - torch.tensor([1.,2.,0.,4.,5.,5.]))**2
#     l.mean().backward()
#     optim.step()
#     print('----------')
#     print(out)
#     print(values)