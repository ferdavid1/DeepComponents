import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from ast import literal_eval

data = pd.read_csv('ImageTopologyDataset.csv')
data = data['ImageStructure'].values[:100]
data = list(map(literal_eval, data))
for x in range(len(data)):
	plt.figure()
	plt.plot(data[x])
	plt.show()
	plt.savefig('number_representation_viz/' + 'viz' + str(x) + '.png')