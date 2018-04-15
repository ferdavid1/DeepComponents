import numpy as np
import pandas as pd
from betti import SimplicialComplex
from PIL import Image
from torchvision.datasets import MNIST

def process_images():
	train_data = MNIST(root = '.', train=True, download=False)	
	df = pd.DataFrame()	
	df['images'] = [np.array(x)*255 for x,y in train_data]
	df['labels'] = [y for x,y in train_data]
	return df['images'].values, df['labels'].values

def find_betti_numbers(image_array): # find betti numbers of image array
	sc = SimplicialComplex(image_array)
	print(sc.betti_number(0))
	
def PersistentHomology():
	data = process_images()
	images, labels = data[0], data[1]
	for image in images:
		find_betti_numbers(image)

PersistentHomology()