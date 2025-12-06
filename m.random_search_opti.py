import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data, vertical_data
import dataset_func as df
nnfs.init();
#hidden layer
class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) #creates array of all zeros
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases;

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True));
        probabilities = exp_values / np.sum(exp_values, axis=1 ,keepdims=True);
        self.output = probabilities

class Loss:
    def calcualte(self,outputs, y):
        sample_losses = self.forward(outputs,y)
        data_loss = np.mean(sample_losses)
        return data_loss;
class Loss_CAtegoricalCrossEntropy(Loss):
    def forward(Self,y_pred, y_true):
        sample = len(y_pred);
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        #scalar values
        if len(y_true.shape) ==1 :
            correct_confidences = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences);
        return negative_log_likelihoods

X,y = vertical_data(100,3)
#create model
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3);
activation2 = Activation_Softmax()
#create loss fucntion
loss_fucntion = Loss_CAtegoricalCrossEntropy()

#variables
lowest_loss = 9999999;
best_dense1_weights = dense1.weights.copy();
best_dense1_biases  = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases  = dense2.biases.copy()

for iteration in range(1000000):
    #genrate new set of weights 
    dense1.weights += 0.5 *np.random.randn(2,3)
    dense1.biases  += 0.5 *np.random.randn(1,3)
    dense2.weights += 0.5 *np.random.randn(3,3)
    dense2.biases  += 0.5 *np.random.randn(1,3)

    #forward pass fo the data
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    #calcualte loss
    loss = loss_fucntion.calcualte(activation2.output, y)
    
    #calculate accuray
    
    predictions = np.argmax(activation2.output, axis =1);
    accuracy = np.mean(predictions==y);
    
    if loss < lowest_loss:
        print('New set of weights founds in iteration', iteration, 'loss' ,loss, 'acc', accuracy)
        best_dense1_weights = dense1.weights.copy();
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()           
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        