import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from ast import literal_eval
from scipy.stats import pearsonr

def pad_component_arrays(x, upper_lim):
    input_x = np.array(x)
    output_x = []
    for array in input_x:
        array += [0]*(upper_lim-len(array))
        output_x.append(array)
    return np.array(output_x)

def corr(data, labels):
	indices = np.where(labels==5)
	indices = indices[0]
	data = pad_component_arrays(data, 750)
	print(len(indices))
	for i in indices[:-1]:	
		print(pearsonr(data[i], data[i+1]))


def plot_morse(data, labels):
	for x in range(len(data)):
		plt.figure()
		# plt.scatter(np.arange(len(data[x])), data[x])
		plt.plot(data[x])
		plt.show()
		# plt.savefig('Visualizations/morse_funcs/' + 'viz_' + 'picture' + str(x+1) + 'digit' + str(labels[x]) + '.png')

def image_draw():
	from PIL import Image 
	from PH import process_images
	data, labels = process_images()
	data, labels = data[:5], labels[:5]
	for c in range(len(data)):
		i = np.array(data[c])
		im = Image.fromarray(i)
		im.save('Visualizations/number_pictures/' +	'picture' + str(c+1) + "digit" + str(labels[c]) + ".png")
		
if __name__ == '__main__':
	data = pd.read_csv('ImageTopologyDataset.csv')
	labels = data['ImageLabels'].values
	data = data['ImageStructure'].values[:51]
	data = list(map(literal_eval, data))
	# plot_morse(data, labels)
	corr(data, labels)
	# image_draw()