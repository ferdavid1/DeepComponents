import torch 
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim

train_data = MNIST(root = '.', train=True, transform=ToTensor(), download=False)
load_train = DataLoader(train_data, batch_size=100, shuffle=True)
test_data = MNIST(root='.', train=False, transform=ToTensor(), download=False)
load_test = DataLoader(test_data, batch_size=100, shuffle=True)
N, D_in, H, D_out = 600, 784, 200, 10

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))
# model = torch.load('firsttry.pt')
loss = torch.nn.CrossEntropyLoss()
learning_rate = 1e3
optim = optim.Adam(model.parameters(), lr=learning_rate)
for i in range(2000):
    for images, labels in load_train:
    	images, labels = Variable(images), Variable(labels)
    	images = images.view(images.size(0), 28*28)
    	y_pred = model(images)
    	lossed = loss(y_pred, labels)
    	print(i, lossed.data[0])
    	optim.zero_grad()
    	lossed.backward()
    	optim.step()
    	for param in model.parameters():
    		param.data -= learning_rate * param.grad.data
	        
    if i%200 == 0:
        print("Error:" + str(loss))
torch.save(model, 'firsttry.pt')
