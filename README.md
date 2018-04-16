# PersistenceRecognition
Image Recognition by Persistent Homology

## Data
- MNIST
- For each image, for each row within the image matrix, a representative sparse vector is created, whose non-zero values are the length of the connected components of the image within that row. 
- For each connected-component representation of each image, an additional abstraction vector is created. 
- In this connected-component-image-vector, for each row of the representation, a row with no connected components is mapped to a 0, and a row with connected components has the lengths of those connected components mapped to an element of the connected-component-image vector with the same value. 
- Thus for each NxN image there exists an N-dimensional vector
- MNIST data is 784-dimensional. 
- For 28x28 data,a log_2_(784) = 9.6 = 10-length vector is needed to accurately represent its features. 
- This process reduces dimensionality of MNIST to 28-D

## Neural Network
- PyTorch
- Very simple, 1-hidden-layer model with ADAM optimization, Cross-Entropy Loss. 
