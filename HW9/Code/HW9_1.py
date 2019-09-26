import numpy as np
import math

def generate_samples():
    m = -1
    c = 1
    s = np.asarray(np.random.uniform(-1,0,1000)).reshape([1,1000])
    y = m*s+c
    print(s.shape, y.shape)
    noise = np.random.normal(0,1,1000)
    return s,y,y+noise

def calculate_loss(target_y, observed_y):
    pass

def MAE(x, target_y, observed_y):
    n = x.shape[0]
    eta = 0.02
    w = np.asarray([0.003,0.001])
    start_loss = np.sum(np.abs(target_y-observed_y))/n
    print("Loss at before gradient descent : ", start_loss)
    for i in range(n):
        dJ = -x[i]*(observed_y[i]-w.dot(x[i]))/abs(observed_y[i]-w.dot(x[i]))
        w = w - eta*dJ
        eta = eta*(3/4)
    end_loss = np.sum(target_y-observed_y)/n
    return w
        

def Huber():
    pass

def Log_cosh():
    pass


x,target_y,observed_y = generate_samples()
# samples = np.append(x,observed_y,axis=0)
# print(samples.shape)
# Loss 1
w = MAE(x, target_y, observed_y)

# Loss 2


# Loss 3