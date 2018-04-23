import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from ast import literal_eval

def plot_morse(data, labels):
	for x in range(len(data)):
		plt.figure()
		plt.plot(data[x])
		# plt.show()
		plt.savefig('Visualizations/morse_funcs/' + 'viz_' + 'picture' + str(x+1) + 'digit' + str(labels[x]) + '.png')


if __name__ == '__main__':
	data = pd.read_csv('ImageTopologyDataset.csv')
	labels = data['ImageLabels'].values
	data = data['ImageStructure'].values[:51]
	data = list(map(literal_eval, data))
	plot_morse(data, labels)