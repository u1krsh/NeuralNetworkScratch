import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data
import dataset_func as df
nnfs.init();


# X = [[1.0, 2.0, 3.0, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]] #(3,4) shape

# inputs = [ 0.2,-1,3.3,-2.7, 1.1, 2.2, -100]
# output = []

# #rectified linear activation func
# for i in inputs:
#     if i> 0:
#         output.append(i)
#     elif  i<= 0:
#         output.append(0);
        
# print(output)
# X,y = df.create_data(100,3)
# X,y = spiral_data(100,3)

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
# layer1 = Layer_Dense(2,5) #whatever you want
# layer2 = Layer_Dense(5,512) 
# activation1 = Activation_ReLU();
# layer1.forward(X)
# activation1.forward(layer1.output)
# print(activation1.output)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

X,y = df.create_data(100,3)
dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3);
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

loss_fucntion = Loss_CAtegoricalCrossEntropy()
loss = loss_fucntion.calcualte(activation2.output,y);
print("Loss: ", loss)
