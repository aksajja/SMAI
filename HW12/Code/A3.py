import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 13
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ x for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("wine.data")

'''
    GD types -
    Batch
    Stochastic
    Stochastic mini-batch

    dJ = -2*sum(xi*(yi-w*xi))   ; number of terms in summation will change based on type of GD
    w` = w - eta*dJ
'''