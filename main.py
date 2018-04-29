import torch 
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torch.optim as optim
import pandas as pd 
import numpy as np
from ast import literal_eval

def pad_component_arrays(x, upper_lim):
    input_x = np.array(x)
    output_x = []
    for array in input_x:
        array += [0]*(upper_lim-len(array))
        output_x.append(array)
    return np.array(output_x)

def load_data(train=True):
    # loading the dataset of morse functions of each number representation
    if train:
        data = pd.read_csv('ImageTopologyDataset.csv')
    else:
        data = pd.read_csv('ImageTopologyTesting.csv')
    train_x = data['ImageStructure'].values
    train_y = Variable(torch.from_numpy(data['ImageLabels'].values), requires_grad=False)
    train_x = list(map(literal_eval, train_x))
    # upper_lim = max([len(x) for x in train_x])
    upper_lim = 51
    train_x = list(map(torch.from_numpy, pad_component_arrays(train_x, upper_lim)))
    train_x = list(map(Variable, [x.float() for x in train_x]))
    return train_x, train_y, upper_lim

train_x, train_y, upper_lim = load_data(train=True)
D_in, H, D_out = upper_lim, 200, 10 
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, 100), torch.nn.ReLU(), torch.nn.Linear(100, D_out))
loss = torch.nn.CrossEntropyLoss()
learning_rate = 1e-2
optim = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# for i in range(2000):
for i in range(10):
    for ind, rep in enumerate(train_x):
        optim.zero_grad()
        rep = rep.view(1, upper_lim)
        y_pred = model(rep)
        lossed = loss(y_pred, train_y[ind])
        print(i, lossed.data[0])
        lossed.backward()
        optim.step()

torch.save(model, 'model.pt')
print('Done Training')

# model = torch.load('model.pt')
# test_x, test_y, ul = load_data(train=False)
# test_x, test_y = test_x[:100], test_y[:100] # test only the first ten images

# # correct = 0
# # total = 0
# for index, rep in enumerate(test_x):
#     rep = rep.view(1, ul)
#     y_pred = model(rep)
#     y_true = test_y[index]
#     _, predicted = torch.max(y_pred.data, 1)
#     print(predicted, y_true)
    # total += test_y.size(0)
    # correct += (predicted == y_true).sum().item()


# print('Accuracy of the network on the 100 test images: %d %%' % (100 * correct / total))
