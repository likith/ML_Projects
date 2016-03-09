import os
import struct
import numpy as np
from array import array

class NeuralNet:

	def __init__(self, hidden_size = 90, alpha = 1.0, activation_in = "sigmoid", mini_batches = 6, epochs = 30, iterations = 90):
		self.input_size = 784
		self.hidden_size = hidden_size
		self.output_size = 10
		self.alpha = alpha
		self.mini_batches_splits = mini_batches
		self.epochs = epochs
		self.iterations = iterations		
		self.weights_0_1 = 2 * np.random.random((self.input_size, self.hidden_size)) - 1
		self.weights_1_2 = 2 * np.random.random((self.hidden_size, self.output_size)) - 1
		
		self.activation_functions = {}		
		self.activation_functions["sigmoid"] = self.sigmoid
		self.activation_functions["tanh"] = self.tanh
		self.activation_primes = {}
		self.activation_primes["sigmoid"] = self.sigmoid_prime
		self.activation_primes["tanh"] = self.tanh_prime

		self.activation = self.activation_functions[activation_in]
		self.activation_prime = self.activation_primes[activation_in]

	def stochastic_gradient_descent(self):
		train_img, train_lbl, test_img, test_lbl = self.load_data("./data")
		img_mini_batches = np.array_split(train_img, self.mini_batches_splits)
		lbl_mini_batches = np.array_split(train_lbl, self.mini_batches_splits)

		mini_batch_size = len(img_mini_batches[0])

		for epoch in np.arange(self.epochs):
			for i, (img_mini_batch, lbl_mini_batch) in enumerate(zip(img_mini_batches,lbl_mini_batches)):
				for j in np.arange(self.iterations):
					activations_lyr_1, activations_lyr_2 = self.forward_propagation(img_mini_batch)
					delta_lyr_1, delta_lyr_2 = self.back_propagation(lbl_mini_batch, activations_lyr_2, activations_lyr_1, activations_lyr_2)

					self.weights_1_2 -= self.alpha/mini_batch_size * np.dot(activations_lyr_1.T, delta_lyr_2)
					self.weights_0_1 -= self.alpha/mini_batch_size * np.dot(img_mini_batch.T, delta_lyr_1)

				accuracy = self.predict(test_img, test_lbl)
				print "accuracy after mini batch %d in epoch %d : %d/%d" %(i+1, epoch, accuracy, len(test_lbl))

	def predict(self, test_img, test_lbl):
		test_activations_1 = self.activation(np.dot(test_img,self.weights_0_1))
		outputs = self.activation(np.dot(test_activations_1, self.weights_1_2))

		accuracy = 0
		for i in np.arange(len(test_lbl)):
			if test_lbl[i] == np.argmax(outputs[i]):
				accuracy += 1

		return accuracy


	def load_data(self, path = '.'):
		train_img_path = os.path.join(path,'train','train-images-idx3-ubyte')
		train_lbl_path = os.path.join(path,'train','train-labels-idx1-ubyte')
		test_img_path = os.path.join(path,'test','t10k-images-idx3-ubyte')
		test_lbl_path = os.path.join(path,'test','t10k-labels-idx1-ubyte')

		print "loading train images and labels..."
		train_images, train_labels = self.load_image_data(train_img_path), self.load_label_data(train_lbl_path)

		print "loading test images and labels..."
		test_images, test_labels = self.load_image_data(test_img_path), self.load_label_data(test_lbl_path)

		train_labels_expand = np.zeros((len(train_labels), self.output_size), dtype = np.float)
		for k in np.arange(len(train_labels)):
			train_labels_expand[k][train_labels[k]] = 1

		return train_images, train_labels_expand, test_images, test_labels

	def load_image_data(self, path):
		with open(path, 'rb') as image_file:
			magic_nbr, size, rows, cols = struct.unpack('>IIII', image_file.read(16))
			img_arr = array("B", image_file.read())
			img_np = np.array(img_arr).reshape(size, rows * cols)
			return img_np

	def load_label_data(self, path):
		with open(path, 'rb') as label_file:
			magic_nbr, size = struct.unpack('>II', label_file.read(8))
			lbl_arr = array("B", label_file.read())
			lbl_np = np.array(lbl_arr).reshape(size, 1)
			return lbl_np

	def forward_propagation(self, X):
		activations_lyr_0 = X
		activations_lyr_1 = self.activation(np.dot(activations_lyr_0, self.weights_0_1))
		activations_lyr_2 = self.activation(np.dot(activations_lyr_1, self.weights_1_2))
		return activations_lyr_1, activations_lyr_2

	def back_propagation(self, out, fp_out, activations_lyr_1, activations_lyr_2):
		error_lyr_2 = (fp_out - out)
		delta_lyr_2 = error_lyr_2 * self.activation_prime(activations_lyr_2)
		error_lyr_1 = delta_lyr_2.dot(self.weights_1_2.T)
		delta_lyr_1 = error_lyr_1 * self.activation_prime(activations_lyr_1)
		return delta_lyr_1, delta_lyr_2

	def sigmoid(self, x):
		return (1/(1 + np.exp(-x)))

	def sigmoid_prime(self, x):
		return x * (1 - x)

	def tanh(self, x):
		return np.tanh(x)

	def tanh_prime(self, x):
		return (1 - x ** 2)


model = NeuralNet()
model.stochastic_gradient_descent()
