import numpy as np
import pandas as pd
from PIL import Image
import networkx as nx 
from scipy.spatial import distance
from itertools import product

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
	structures = []
	df['ImageStructure'] = np.array([morse(image) for image in images])
	
	df['ImageLabels'] = labels 
	
	# df.to_csv('ImageTopologyTesting.csv', index=False)
	df.to_csv('ImageTopologyDataset.csv', index=False)

class SimplicialComplex:
	def __init__(self, simplices=[]):
		self.simplices = simplices
		self.face_set = self.faces()

	def faces(self):
		faceset = set()
		for simplex in self.simplices:
			numnodes = len(simplex)
			for r in range(numnodes, 0, -1):
				for face in combinations(simplex,r):
					faceset.add(face)
		return faceset

	def n_faces(self, n):
 		return filter(lambda face: len(face)==n+1, self.face_set)

class VietorisRipsComplex(SimplicialComplex):
	def __init__(self, points, epsilon, labels=None, distfnc=distance.euclidean):
		self.pts = points
		self.epsilon = epsilon
		self.labels = range(len(self.pts))
		self.distfnc = distfnc
		self.network = self.construct_network(self.pts, self.labels, self.epsilon, self.distfnc)

	def construct_network(self, points, labels, epsilon, distfnc):
		g = nx.Graph()
		g.add_nodes_from(labels)
		zips = list(zip(points, labels))
		for pair in product(zips, zips):
			if pair[0][1] != pair[1][1]: # if not the same point
				dist = distfnc(pair[0][0], pair[1][0])
				if dist < epsilon:
					g.add_edge(pair[0][1], pair[1][1])
		return g

def PersistentHomology(morse_function_values, epsilon): # epsilon is the neighborhood radius value for the Vietoris-Rips Complex
	import matplotlib.pyplot as plt 
	vr = VietorisRipsComplex(morse_function_values, epsilon)
	G = vr.network
	nx.draw(G, with_labels=True)
	plt.show()

def test_PH():	
	from ast import literal_eval
	data = pd.read_csv('ImageTopologyDataset.csv')
	data = data['ImageStructure'].values[:10]
	data = list(map(literal_eval, data))

	for x in range(len(data)):
		PersistentHomology(data[x])

# make_dataset()
test_PH()