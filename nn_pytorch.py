import torch
from torch.autograd import Variable
import torch.optim as optim
N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
	y_pred = model(x)
	loss = loss_fn(y_pred, y)
	print(t, loss.data[0])
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	for param in model.parameters():
		param.data -= learning_rate * param.grad.data