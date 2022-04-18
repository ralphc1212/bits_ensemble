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


def quantization_function(module, weight_quantization=True, act=None, shape=None):

    if weight_quantization:
        # use the params for weight quantization
        param_idx = 0
        # U is of size NxD, D=D1xD2
        target_matrix = module.U
    else:
        # use the params for activation quantization
        param_idx = 1
        # act is of size BxNxD2
        # better work on NxD2, rather than depend on batch
        # thres is of size D2, no repeat,
        # transmat needs N for onehot
        target_matrix = act

    module.npartitions[param_idx] = [0] * (module.maxBWPow - 1)

    beta = torch.max(target_matrix)
    alpha = torch.min(target_matrix)
    # group_counts = [0] * (maxBWPow - 1)

    s = None
    for idx, deno in enumerate(module.res_denos[:module.maxBWPow]):
        if s is None:
            s = (beta - alpha) / deno
            vals = (s * module.round(target_matrix/ s)).unsqueeze(0)
        else:
            s = s / deno
            res_errer = target_matrix - vals.sum(0)

            if not weight_quantization:
                sorted_err, idx_maps = torch.sort(res_errer, dim=1)
                transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=module.N).float().detach()
                delta_sorted_err = sorted_err[:,1:,:] - sorted_err[:,:-1,:]
                delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err,dim=0)[0]
                mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(module.thres_mean[param_idx].repeat_interleave(int(delta_sorted_err.shape[-1]/module.thres_mean[param_idx].shape[0]))), 0.01)
                module.npartitions[param_idx][idx - 1] += torch.round(mean_split_point).mean(dim=(0,2)).sum().item()
                round_split = torch.cat([module.zeroD[param_idx].expand(mean_split_point.shape[0], -1, mean_split_point.shape[2]), mean_split_point],dim=1)
                clustering_vec = rbf_wmeans(round_split.cumsum(dim=1).unsqueeze(3), module.rbf_mu, module.rbf_sigma, module.eps)
                inner_grouped = torch.einsum('abcd, acd -> abc', clustering_vec, (sorted_err.unsqueeze(3) * clustering_vec).sum(dim=1) / (clustering_vec.sum(dim=1)+module.eps))
                grouped = torch.einsum('dcb, dcba -> dab', inner_grouped, transfer_matrix)

            else:
                if module.update_partition:
                    sorted_err, idx_maps = torch.sort(res_errer, dim=0)
                    module.transfer_matrix = torch.nn.functional.one_hot(idx_maps, num_classes=module.N).float().detach()
                    delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
                    delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
                    mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(module.thres_mean[param_idx].repeat_interleave(int(delta_sorted_err.shape[-1] / module.thres_mean[param_idx].shape[0]))), 0.01)
                    module.split_point = mean_split_point.detach()
                    module.npartitions[param_idx][idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()
                    round_split = torch.cat([module.zeroD[param_idx], mean_split_point])
                else:
                    sorted_err = torch.einsum('ab, cba -> cb', res_errer, module.transfer_matrix)
                    module.npartitions[param_idx][idx - 1] += torch.round(module.split_point).mean(dim=1).sum().item()
                    round_split = torch.cat([module.zeroD[param_idx], module.split_point])

                clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), module.rbf_mu, module.rbf_sigma, module.eps)
                inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0) + module.eps))
                grouped = torch.einsum('cb, cba -> ab', inner_grouped, module.transfer_matrix)

            quantized = s * module.round(grouped / s)
            vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)
    return vals

            # if module.update_partition[param_idx]:
            #     if weight_quantization:
            #         sorted_err, idx_maps = torch.sort(res_errer, dim=0)
            #     else:
            #         sorted_err, idx_maps = torch.sort(res_errer, dim=1)

            #     module.transfer_matrix[param_idx] = torch.nn.functional.one_hot(idx_maps, num_classes=module.N).float().detach()

            #     # transfered_error = torch.einsum('ab, cba -> cb', res_errer, self.transfer_matrix)
            #     if weight_quantization:
            #         delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
            #         delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
            #     else:
            #         delta_sorted_err = sorted_err[:,1:,:] - sorted_err[:,:-1,:]
            #         delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err,dim=0)[0]

            #     mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(module.thres_mean[param_idx].repeat_interleave(int(delta_sorted_err.shape[-1]/module.thres_mean[param_idx].shape[0]))), 0.01)

            #     ### May need let the partitioning shared in a channel.
            #     module.split_point[param_idx] = mean_split_point.detach()
            #     if weight_quantization:
            #         module.npartitions[param_idx][idx - 1] += torch.round(mean_split_point).mean(dim=1).sum().item()
            #         round_split = torch.cat([module.zeroD[param_idx], mean_split_point])
            #     else:
            #         module.npartitions[param_idx][idx - 1] += torch.round(mean_split_point).mean(dim=(0,2)).sum().item()
            #         round_split = torch.cat([module.zeroD[param_idx].expand(mean_split_point.shape[0], -1, mean_split_point.shape[2]), mean_split_point],dim=1)

            # else:
            #     if weight_quantization:
            #         sorted_err = torch.einsum('ab, cba -> cb', res_errer, module.transfer_matrix[param_idx])
            #         module.npartitions[param_idx][idx - 1] += torch.round(module.split_point[param_idx]).mean(dim=1).sum().item()
            #         round_split = torch.cat([module.zeroD[param_idx], module.split_point[param_idx]])
            #     else:
            #         if not module.update_partition[param_idx]:
            #             print(res_errer.shape)
            #             print(module.transfer_matrix[param_idx].shape)
            #         sorted_err = torch.einsum('abc, dbca -> dbc', res_errer, module.transfer_matrix[param_idx])
            #         module.npartitions[param_idx][idx - 1] += torch.round(module.split_point[param_idx]).mean(dim=(0,2)).sum().item()
            #         round_split = torch.cat([module.zeroD[param_idx].expand(module.split_point[param_idx].shape[0], -1, module.split_point[param_idx].shape[2]), module.split_point[param_idx]],dim=1)

            #     # # stt-round
            #     # round_split = torch.cat([self.zeroD, self.round(self.split_point)])
            #     # soft
            # if weight_quantization:
            #     clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), module.rbf_mu, module.rbf_sigma, module.eps)
            #     inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+module.eps))
            #     grouped = torch.einsum('cb, cba -> ab', inner_grouped, module.transfer_matrix[param_idx])
            # else:
            #     clustering_vec = rbf_wmeans(round_split.cumsum(dim=1).unsqueeze(3), module.rbf_mu, module.rbf_sigma, module.eps)
            #     inner_grouped = torch.einsum('abcd, acd -> abc', clustering_vec, (sorted_err.unsqueeze(3) * clustering_vec).sum(dim=1) / (clustering_vec.sum(dim=1)+module.eps))

            #     grouped = torch.einsum('dcb, dcba -> dab', inner_grouped, module.transfer_matrix[param_idx])

    #         quantized = s * module.round(grouped / s)
    #         vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)
    #         # if not weight_quantization:
    #         #     for i in range(act.shape[0]):
    #         #         print('--------------------',i)
    #         #         print(act[i])
    #         #         print(vals.sum(0)[i])
    #         #     exit()
    # return vals


class dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2, bw=3, quantize=True):
        super(dense_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2
        D = int(D1 * D2)
        self.D = D
        self.U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

        self.res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)

        self.round = get_round()
        self.maxBWPow = bw

        self.zeroD = nn.ParameterList([nn.Parameter(torch.zeros(1, self.D), requires_grad=False), nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)])
        self.thres_mean = nn.ParameterList([nn.Parameter(torch.tensor([-1.3] * D2), requires_grad=True),nn.Parameter(torch.tensor([-1.3] * D2), requires_grad=True)])
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.npartitions = [None, None]
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)
        self.repeatnum = D1

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):

        vals = quantization_function(self, weight_quantization=True)
        w = vals.sum(0).view(self.N, self.D1, self.D2)

        act = torch.einsum('bnd, ndl -> bnl', x, w)

        quantized_act = quantization_function(self, weight_quantization=False, act=act).sum(0)

        if torch.sum(torch.isnan(quantized_act)) != 0:
            print('act nan')
            print(act)
            exit()

        return quantized_act

class Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, N=5, bw=2, quantize=True, first=False):
        super(Conv2d_res_bit_tree_meanvar_freeze_fine_partition_wkernel_wact_baens, self).__init__()

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

        self.zeroD = nn.ParameterList([nn.Parameter(torch.zeros(1, self.D), requires_grad=False), nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)])
        self.thres_mean = nn.ParameterList([nn.Parameter(torch.tensor([-1.3] * out_channels), requires_grad=True), nn.Parameter(torch.tensor([-1.3] * out_channels), requires_grad=True)])
        self.transfer_matrix = None
        self.split_point = None
        self.update_partition = True
        self.npartitions = [None, None]
        self.eps = nn.Parameter(torch.tensor([10 ** (-16)]), requires_grad=False)
        self.repeatnum = int(self.in_channels * self.kernel_size * self.kernel_size)

        self.rbf_mu = nn.Parameter(torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0), requires_grad=False)
        self.rbf_sigma = nn.Parameter(torch.tensor([0.1]), requires_grad=False)

        self.temperature = TEMPERATURE
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):
        if not self.first:
            x = x.view(int(x.shape[0]/self.N), int(self.N * x.shape[1]), *x.shape[2:])

        # betas, _ = torch.max(self.U, dim=1)
        # alphas, _ = torch.min(self.U, dim=1)
        vals = quantization_function(self, weight_quantization=True)

        w = vals.sum(0).view(int(self.out_channels * self.N), int(self.in_channels), self.kernel_size, self.kernel_size)

        # x should be of the size (sub-batch-size , (self.N * in_channels) , kernel_size, kernel_size)
        act = torch.nn.functional.conv2d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.N)
        quantized_act = quantization_function(self, weight_quantization=False, act=act.view(act.shape[0], self.N, -1))
        act = quantized_act.sum(0).view(int(act.shape[0] * self.N), int(act.shape[1] / self.N), *act.shape[2:])

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