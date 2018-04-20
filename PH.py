import numpy as np
import pandas as pd
from PIL import Image
import networkx as nx 
from scipy.spatial import distance
from itertools import product
from ast import literal_eval

def process_images():
	from torchvision.datasets import MNIST
	train_data = MNIST(root = '.', train=True, download=False)	
	test_data = MNIST(root='.', train=False, download=False)
	data = train_data
	# data = test_data
	df = pd.DataFrame()	
	df['images'] = [np.array(x)*255 for x,y in data]
	df['labels'] = [y for x,y in data]
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

def make_dataset():
	data = process_images()
	images, labels = data[0], data[1]
	df = pd.DataFrame()
	morsed = [morse(image) for image in images]
	final_structures = []
	for struct in morsed:
		struct = np.array(struct)
		final_structures.append(struct[struct!=0])
	df['ImageStructure'] = final_structures
	df['ImageLabels'] = labels 
	
	# df.to_csv('ImageTopologyTesting.csv', index=False)
	df.to_csv('ImageTopologyDataset.csv', index=False)

make_dataset()