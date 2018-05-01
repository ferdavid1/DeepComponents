from PH import process_images, morse
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

def generate():
	imgs, labels = process_images()
	imgs = imgs[:51]
	imgs = [morse(list(image)) for image in imgs] # morsify, but dont remove 0-value structure
	new_imgs = []
	for ind,i in enumerate(imgs):
		subarray = []
		for indx, j in enumerate(i):
			if j != 0:
				for x in range(j):
					rand = np.random.randint(1,256)
					subarray.append(rand)
			else:
				subarray.append(0)
		new_imgs.append(subarray)

	for ind,i in enumerate(new_imgs):
		if len(i) != 784:
			diff = 784 - len(i)
			for x in range(diff):
				i.append(0)
		i = np.asarray(i, np.uint8).reshape((28,28))
		im = Image.fromarray(i)
		im.save("Visualizations/generated/Representation"+ str(ind) + "Digit" + str(labels[ind]) + ".png")

generate()