import torch 
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd 
import numpy as np
from ast import literal_eval

def pad_component_arrays(x):
    # across images, the max connected_components size is 750, 
    # upper_lim = max([len(x) for x in train_x]) this is 750
    upper_lim = 750
    output_x = np.array([np.pad(np.array(array, dtype='int32'), 750-len(array), mode='constant', constant_values=0) for array in x], dtype='int32')
    return output_x

def load_train_data():
    data = pd.read_csv('ImageTopologyDataset.csv')
    train_x = data['ImageStructure'].values
    train_x = list(map(literal_eval, train_x))
    train_x = torch.from_numpy(pad_component_arrays(train_x))
    return train_x

train_x = load_train_data()
# test_data = MNIST(root='.', train=False, transform=ToTensor(), download=False)
# load_test = DataLoader(test_data, batch_size=100, shuffle=True)
N, D_in, H, D_out = len(train_x), 94, 47, 10 
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))
# model = torch.load('firsttry.pt')
loss = torch.nn.CrossEntropyLoss()
learning_rate = 1e3
optim = optim.Adam(model.parameters(), lr=learning_rate)
for i in range(2000):
    y_pred = model(train_x)
    print('a')
# 	lossed = loss(y_pred, labels)
# 	print(i, lossed.data[0])
# 	optim.zero_grad()
# 	lossed.backward()
# 	optim.step()
# 	for param in model.parameters():
# 		param.data -= learning_rate * param.grad.data
	        
#     if i%200 == 0:
#         print("Error:" + str(loss))
# torch.save(model, 'firsttry.pt')
