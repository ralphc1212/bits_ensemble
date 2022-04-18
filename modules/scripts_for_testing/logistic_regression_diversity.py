import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class LogisticRegression(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LogisticRegression, self).__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, x):
		outputs = self.linear(x)
		return outputs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=True, transform=transforms.Compose([
                   transforms.RandomCrop(28, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
test_dataset = torchvision.datasets.mnist.MNIST(root='~/data', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=192, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=192, shuffle=False)