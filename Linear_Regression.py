# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import random

data = pd.read_csv("data/Admission_Predict.csv")

### Get data from CSV file
arr = data.to_numpy()
arr2 = arr[:,1:]

shuffled_data = arr2.T[:, np.random.permutation(arr2.T.shape[1])].T ### Shuffle data
training_data = shuffled_data[0:300] ### Get data used for training
testing_data = shuffled_data[300:]   ### Get data used for testing

training_input = training_data[:,:-1]   ### Input training data
training_output = training_data[:,-1:]   ### Output training data
testing_input = testing_data[:,:-1]
testing_output = testing_data[:,-1:]


class Linear_Regression():
    def __init__(self,x,w,y,learning_rate = 0.0001):
        self.X = x
        self.y = y
        self.w = w
        self.learning_rate = learning_rate
        self.N = len(self.y)
    def loss(self):
        sum = float(0)
        sum = np.linalg.norm((self.X .dot(self.w) - self.y))
        return sum**2/(2*self.N)
    def grad(self):
        return (self.X.T .dot(self.X .dot(self.w) - self.y)) / self.N
    def grad_Descent(self):
        it = float(0)
        lst = list()
        while it < 100000:
            if self.loss() < 4e-3:
                break
            lst.append(self.loss())
            self.w = self.w - self.learning_rate*self.grad()
            it = it + 1
        return lst,it

w = 9*np.ones((7,1))
trainer = Linear_Regression(training_input,w,training_output,0.00001)
lst,it = trainer.grad_Descent()
print("vector W after training: ")
print(trainer.w)
print("Loss: %.8f after %.0f iterations " %(trainer.loss(),it))

tester = Linear_Regression(testing_input, trainer.w, testing_output)
print("Loss: %.8f" %(tester.loss()))