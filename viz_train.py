import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from ast import literal_eval

data = pd.read_csv('ImageTopologyDataset.csv')
labels = data['ImageLabels'].values
data = data['ImageStructure'].values[:20]
data = list(map(literal_eval, data))
for x in range(len(data)):
	plt.figure()
	plt.plot(data[x])
	# plt.show()
	plt.savefig('Visualizations/morse_funcs/' + 'viz_' + 'picture' + str(x) + 'digit' + str(labels[x]) + '.png')