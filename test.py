import torch 
from torch.autograd import Variable
import numpy as np 
tot = np.array([np.array([1,2]), np.array([2,3])])
print(torch.from_numpy(tot))