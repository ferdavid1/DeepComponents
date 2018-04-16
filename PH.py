import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import MNIST

def process_images():
	train_data = MNIST(root = '.', train=True, download=False)	
	df = pd.DataFrame()	
	df['images'] = [np.array(x)*255 for x,y in train_data]
	df['labels'] = [y for x,y in train_data]
	return df['images'].values, df['labels'].values

def morse(image_array): # plot image values as a signal, turned into a morse function 
	connected_components = []
	for x in image_array:
		cc = 0
		ccs = []
		for value in x:
			if value != 0:
				cc += 1
			elif value == 0 and cc > 0:
				ccs.append(cc)
				cc = 0
			else:
				ccs.append(0)
		connected_components.append(ccs)
	new_connected = []
	for c in connected_components:
		if c == []:
			new_connected.append(0)
		else:
			for x in c:
				new_connected.append(x)
	return new_connected

def PersistentHomology():
	data = process_images()
	images, labels = data[0], data[1]
	df = pd.DataFrame()
	structures = []
	df['ImageStructure'] = np.array([morse(image) for image in images])
	df['ImageLabels'] = labels 
	
	df.to_csv('ImageTopologyDataset.csv', index=False)

PersistentHomology()