import torch 
from torch.autograd import Variable
from torchvision.datasets import MNIST
import torch.optim as optim
import torch.utils.data as data_utils
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
        shuff = True
    else:
        data = pd.read_csv('ImageTopologyTesting.csv')
        shuff = False
    train_x = data['ImageStructure'].values
    train_y = data['ImageLabels'].values
    train_x = list(map(literal_eval, train_x))
    train_y = torch.from_numpy(train_y)
    upper_lim = 51 # if the small representation
    # upper_lim = 209 # if medium representation (inbetween zeros)
    # upper_lim = 750 # if the big representation
    # print(max([len(x) for x in train_x]))
    train_x = torch.from_numpy(pad_component_arrays(train_x, upper_lim))
    train = data_utils.TensorDataset(train_x, train_y)
    train_loader = data_utils.DataLoader(train, batch_size=100, num_workers=5, shuffle=shuff)
    return train_loader, upper_lim

train, upper_lim = load_data(train=True)
D_in, H, D_out = upper_lim, 200, 10 
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, 100), torch.nn.ReLU(), torch.nn.Linear(100, D_out))
loss = torch.nn.CrossEntropyLoss()
learning_rate = 1e-2
optim = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

print("Started Training")
for i in range(25):
    for ind, (reps,labels) in enumerate(train):
        reps, labels = Variable(reps.float(), requires_grad=False), Variable(labels, requires_grad=False)
        optim.zero_grad()
        y_pred = model(reps)
        lossed = loss(y_pred, labels)
        if ind%2000 == 0:
            print(i, lossed.data[0])
        lossed.backward()
        optim.step()

torch.save(model, 'model.pt')
print('Done Training')

model = torch.load('model.pt')
test, ul = load_data(train=False)

correct = 0
total = 0
print("Started Testing")
for index, (reps, labels) in enumerate(test):
    reps, labels = Variable(reps.float(), requires_grad=False), Variable(labels, requires_grad=False)
    y_pred = model(reps)
    _, predicted = torch.max(y_pred.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
print("Done Testing")

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
