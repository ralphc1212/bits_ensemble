import torch
import time
import torch.nn.functional as F

N = 4
out_ = 3
in_ = 2
kern = 1

ori = torch.arange(int(N*in_)) + 1
x = ori.view(1, int(N*in_), 1, 1).float()
print(x)
print(x.shape)

print(x.view(1, N, -1))
exit()
mul = 10 * torch.ones(out_ * N * in_)
mul = mul.cumprod(dim=0) / 10.
mul = mul.view(int(out_ * N), in_, 1, 1)

print(mul)
print(mul.shape)
output = F.conv2d(x, mul, bias=None,  groups=N)
print(output)
output = output.view(N, out_, 1, 1)
# stride=1,padding=0,dilation=0,

output = output.view(1, int(N*out_), 1, 1)
print('-------')
print(output)
print(output.shape)
ck = torch.chunk(output, N, dim=1)
print(ck)
print(ck[0].shape)
gp = torch.cat(ck, dim=1)
print(gp)
print(gp.shape)

print('-------')
a = torch.nn.GroupNorm(N, 12)

o_ = a(output)
print(o_)
# trans = output.view(int(N * out_), in_, kern, kern)
# print(trans)
# print(trans.shape)


# a = torch.arange(5).float()
# ri = torch.tensor([0,1,4,2,3])
# non = torch.zeros(5)
# x = non.index_put(indices=[ri], values=a)
# print(x)
# print(a)
