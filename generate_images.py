import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

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

def generate():
	imgs, labels = process_images()
	imgs = imgs[:100]
	imgs = [morse(list(image)) for image in imgs] # morsify, but dont remove 0-value structure
	for ind,i in enumerate(imgs):
		for indx, j in enumerate(i):
			if j != 0:
				print(j)
				for x in range(j):
					imgs[ind].insert(indx+x, np.random.randint(1,255))
			else:
				pass
	print(imgs)
	for ind,i in enumerate(imgs):
		im = Image.fromarray(i)
		im.save('Visualizations/generated/Representation'+ str(ind) + 'Digit' + str(labels[ind]))

generate()