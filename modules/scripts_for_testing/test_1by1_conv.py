import torch
import time
import torch.nn.functional as F

mode = 'conv'

if mode is 'linear':
	x = torch.rand([1024, 4096]).cuda()

	# l1 = torch.nn.Linear(256, 4096, bias=False).cuda()
	l2 = torch.nn.Linear(4096, 256, bias=False).cuda()
else:
	x = torch.rand([1024, 16, 256]).cuda()

	# l1 = torch.nn.Conv1d(256, 4096, 1, padding=0, bias=False).cuda()
	l2 = torch.nn.Conv1d(16, 16, 16, 16, padding=0, bias=False).cuda()
	print(l2.weight.numel())

start = time.time()
for i in range(1000):
	l2(x)

print(time.time() - start)
print(l2(x).shape)
