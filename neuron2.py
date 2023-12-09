import numpy as np
np.random.seed(0)

X=[[1,2,3,4],
   [4,7,2,6],
   [2,3,9,8]]

class layerDense:
    def __init__(self,n_inputs,n_neuron):
        self.weights = 0.10*np.random.randn(n_inputs,n_neuron)
        self.biases = np.zeros((1,n_neuron))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.biases

layer1=layerDense(4,5)
layer2=layerDense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)