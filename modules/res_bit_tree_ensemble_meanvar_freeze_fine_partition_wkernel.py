import torch
import torch.nn as nn

# only weight quantization, without decomposition

TEMPERATURE = 0.01
THRES = 0.9

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
    return torch.exp(- torch.abs(x - means + eps) / (2 * sigma ** 2))

class dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D

        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.Parameter(torch.zeros(1, self.D), requires_grad=False)
        self.thres_mean = nn.Parameter(torch.tensor([-1.3] * D2), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        # wQ = self.quant(self.U)
        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)
        # group_counts = [0] * (maxBWPow - 1)
        
        # s is shared across all values
        # rounding would give the acutal bits
        # a network load the weights
        # grouping is conducted for all three matrices
        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_mean.repeat_interleave(self.D1)), 0.01)
                    ### May need let the partitioning shared in a channel.
                    self.split_point = mean_split_point.detach()
                    self.npartitions[idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(mean_split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD, mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    self.npartitions[idx - 1] += torch.round(self.split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(self.split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD, self.split_point])

                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), self.rbf_mu, self.rbf_sigma, self.eps)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+self.eps))

                grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act


class Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_baens, self).__init__()

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
        
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        # self.quant = res_quant(maxBWPow=3)
        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.Parameter(torch.zeros(1, self.D), requires_grad=False)
        self.thres_mean = nn.Parameter(torch.tensor([-1.3] * out_channels), requires_grad=True)
        self.thres_var  = nn.Parameter(torch.tensor([-2.197]), requires_grad=True)
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        self.npartitions = [0] * (self.maxBWPow - 1)

        beta = torch.max(self.U)
        alpha = torch.min(self.U)

        s = None
        for idx, deno in enumerate(self.res_denos[:self.maxBWPow]):
            if s is None:
                s = (beta - alpha) / deno
                vals = (s * self.round(self.U / s)).unsqueeze(0)
            else:
                s = s / deno
                res_errer = self.U - vals.sum(0)

                if self.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    self.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=self.N).float().detach()

                    # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(self.thres_mean.repeat_interleave(int(self.in_channels * self.kernel_size * self.kernel_size))), 0.01)
                    ### May need let the partitioning shared in a channel.
                    self.split_point = mean_split_point.detach()
                    self.npartitions[idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(mean_split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD, mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)

                    self.npartitions[idx - 1] += torch.round(self.split_point).mean(dim=1).sum().item()

                    # # stt-round
                    # round_split = torch.cat([self.zeroD, self.round(self.split_point)])
                    # soft
                    round_split = torch.cat([self.zeroD, self.split_point])

                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), self.rbf_mu, self.rbf_sigma, self.eps)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+self.eps))

                grouped = torch.einsum('cb, cba -> ab', inner_grouped, self.transfer_matrix)
                quantized = s * self.round(grouped / s)

                vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

        w = vals.sum(0).view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

        # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
        act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)
        act = act.view(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])
        
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