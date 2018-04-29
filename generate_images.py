from PH import process_images, morse
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

def generate():
	imgs, labels = process_images()
	imgs = imgs[:10]
	imgs = [morse(list(image)) for image in imgs] # morsify, but dont remove 0-value structure
	for ind,i in enumerate(imgs):
		for indx, j in enumerate(i):
			if j != 0:
				for x in range(j):
					imgs[ind].append(np.random.randint(1,255))
			else:
				imgs[ind].append(j)
	for ind,i in enumerate(imgs):
		im = Image.fromarray(i)
		im.save('Visualizations/generated/Representation'+ str(ind) + 'Digit' + str(labels[ind]))

generate()