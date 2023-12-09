# Hardcoding Softmax function
# Combination of Exponentiation and Normalization is Softmax Function

import math
import numpy as np
layer_ops=[1.3,4.3,1.02,2.68]

E=math.e
exp_values=np.exp(layer_ops)
'''
for i in layer_ops:
    exp_values.append(E*i)
print(exp_values)
'''
norm_values=exp_values/np.sum(exp_values)
print(norm_values)
'''
norm_base=sum(exp_values)
norm_values=[]
for j in exp_values:
    norm_values.append(j/norm_base)
print(norm_values)
print(sum(norm_values))
'''
# Softmax function working on batches of Neurons Outputs
layer_opsB=[[1.3,4.3,1.02],
           [1.5,4.2,6.02],
           [9.3,1.3,5.62]]

exp_vals=np.exp(layer_opsB)
print(np.sum(layer_opsB,axis=1, keepdims=True))
norm_vals=exp_vals/np.sum(exp_vals,axis=1, keepdims=True)
print(norm_vals)