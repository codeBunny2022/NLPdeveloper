# Categorical cross entropy analysis includes the formula
'''Categorical cross entropy 
Loss=negative sum of product of target value and log of predicted value
for each value in prediction'''
import math
softmax_opts=[0.7,0.2,0.1]
target_opts=[1,0,0]

loss=-(math.log(softmax_opts[0])*target_opts[0]+
       math.log(softmax_opts[1])*target_opts[1]+
       math.log(softmax_opts[2])*target_opts[2])

print(loss)