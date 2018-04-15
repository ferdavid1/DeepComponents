import numpy as np
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader

def process_images():

	train_data = MNIST(root = '.', train=True, download=False)	
	for x in train_data:
		x[0].show()
	image_final = np.array([Image.open(x) for x,y in load_train])
	return image_final

def find_betti_numbers():
	pass
def PersistentHomology():
	data = process_images()
	for batch in data:
		for image in batch:
			image = image.numpy()
			print(image)
			a = Image.fromarray(image*255)
			a.show()
			# find_betti_numbers()
import matplotlib.pyplot as plt 
from PIL import Image
PersistentHomology()