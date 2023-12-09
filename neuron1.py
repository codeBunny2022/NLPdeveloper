# Coding a single neuron
val=[1,2,3]
weights=[9,6,4]
bias=2
'''
result=0
for i in range(len(val)):
    result=result+val[i]*weights[i]
print(result+bias)
'''
import numpy as np
print(np.dot(val,weights)+bias)

# Coding three neurons
values=[1,2,3,4]
weightsList=[[9,6,4,5],
             [9,1,8,5],
             [9,7,3,8]]
biasesList=[5,7,3]

print(np.dot(weightsList,values)+biasesList)
'''
layerOutputs=[] 
for neuwts,neubias in zip(weightsList,biasesList):
    neuoutput = 0
    for neuval,weight in zip(values,weightsList):
        neuoutput+=neuval*weight
    neuoutput+=neubias
    layerOutputs.append(neuoutput)

print(layerOutputs)
'''