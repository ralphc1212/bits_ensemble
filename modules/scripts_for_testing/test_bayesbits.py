import torch
import torch.nn as nn

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
rbf_sigma = 0.1

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

        rounded = ROUND(res_errer / s)
        print('round................\n', rounded, '\n', s)

        quantized = s * rounded

        vals = torch.cat([vals, quantized.unsqueeze(0)], dim=0)

w = vals.sum(0)

print(vals)

print(U)

print(w)
