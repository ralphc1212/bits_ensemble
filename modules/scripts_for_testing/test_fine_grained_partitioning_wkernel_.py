import torch
import torch.nn as nn

# eps = 10 ** (-32)


# sorted_err = torch.tensor([[3,2],[5,6],[7,8]]).float()
# clustering_vec = torch.tensor([[[1,0,0],[1,0,0]],[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]]]).float()

# print(clustering_vec)
# inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+eps))

# print(inner_grouped)

# exit()

# only weight quantization, without decomposition

TEMPERATURE = 1.
THRES = 0.9

def get_round():
    class round(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(input)
            # return torch.floor(input)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return round().apply

def df_lt(a, b, temp):
    # if a > b
    return torch.sigmoid((a - b) / temp)

def rbf_wmeans(x, means, sigma):
    return torch.exp(- torch.sqrt((x - means) ** 2 + eps) / (2 * sigma ** 2))


res_denos = nn.Parameter(torch.tensor([2**2-1, 2**2+1, 2**4+1, 2**8+1, 2**16+1]), requires_grad=False)
thres_mean = nn.Parameter(torch.tensor([-1.2]*3), requires_grad=True)
thres_var  = nn.Parameter(torch.tensor([-3.2]), requires_grad=True)

N = 4
D1 = 2
D2 = 3
D = int(D1*D2)

U = nn.Parameter(torch.normal(0, 1, (N, D)), requires_grad=True)

beta = torch.max(U)
alpha = torch.min(U)
buf = torch.zeros(D)

rbf_mu = torch.tensor([0., 1., 2., 3.]).unsqueeze(dim=0)
rbf_sigma = 0.6

ROUND = get_round()

maxBWPow = 3

zeros = torch.zeros(1, D)

eps = 10 ** (-32)

s = None
for idx, deno in enumerate(res_denos[:maxBWPow]):
    if s is None:
        s = (beta - alpha) / deno
        rounded = ROUND(U / s)
        print('round................\n', rounded, '\n', s)
        vals = (s * rounded).unsqueeze(0)
    else:
        s = s / deno
        res_errer = U - vals.sum(0)
        # print('------------------')
        # print('residual err: \n', res_errer)
        sorted_err, idx_maps = torch.sort(res_errer, dim=0)
        # print('sorted err: \n', sorted_err)
        # print('index map: \n', idx_maps)

        transfered_error = torch.einsum('ab, cba -> cb', res_errer, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())
        # print('transfered error: \n', transfered_error)

        delta_sorted_err = sorted_err[1:,:] - sorted_err[:-1,:]
        delta_sorted_err = delta_sorted_err / torch.max(delta_sorted_err)
        # print(delta_sorted_err.shape)
        # print(thres_mean.repeat_interleave(2).shape)
        mean_split_point = df_lt(delta_sorted_err, torch.sigmoid(thres_mean.repeat_interleave(2)), 0.01)
        ### May need let the partitioning shared in a channel.
        print(mean_split_point)
        print(mean_split_point.shape)

        # var_split_point = df_lt(torch.sigmoid(thres_var), delta_sorted_err.var(dim=1), 0.01)
        # print(mean_split_point)
        # print(torch.round(mean_split_point))
        # print(torch.round(mean_split_point).mean(dim=1).sum().item())
        # round_split = torch.round(mean_split_point)
        round_split = torch.cat([zeros, mean_split_point])
        # print(round_split)
        # print((round_split.cumsum(dim=0)))
        # # print(torch.exp(-torch.sqrt((round_split.cumsum(dim=0).unsqueeze(2) - rbf_mu) ** 2 + eps) / ( 2 * rbf_sigma **2 )))
        # print(rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), rbf_mu, rbf_sigma))
        clustering_vec = rbf_wmeans(round_split.cumsum(dim=0).unsqueeze(2), rbf_mu, rbf_sigma)

        print('&&&&&&&&&&&&&&&&')
        print(round_split)
        print(clustering_vec)
        clustering_vec = ROUND(clustering_vec)
        # print('clustering_vec\n', clustering_vec)
        # print((sorted_err.unsqueeze(2) * clustering_vec).shape)
        # print((sorted_err.unsqueeze(2) * clustering_vec))
        # print(sorted_err.shape)
        # print(clustering_vec.shape)
        inner_grouped = torch.einsum('abc, bc -> ab', clustering_vec, (sorted_err.unsqueeze(2) * clustering_vec).sum(dim=0) / (clustering_vec.sum(dim=0)+eps))
        print(clustering_vec)
        print(sorted_err)
        print(inner_grouped)
        print(ROUND(inner_grouped / s))
        print('&&&&&&&&&&&&&&&&')

        grouped = torch.einsum('cb, cba -> ab', inner_grouped, torch.nn.functional.one_hot(idx_maps, num_classes=4).float())

        # print('output: \n', grouped)
        # print(res_errer)
        # exit()
        rounded = ROUND(grouped / s)
        print('round................\n', rounded, '\n', s)

        quantized = s * rounded

        vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

w = vals.sum(0)

# print(vals)

# print(U)

# print(w)
