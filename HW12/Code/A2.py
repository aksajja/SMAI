import numpy as np
X = np.array([
    [1,1,1],
    [-1,-1,1],
    [2,2,1],
    [-2,-2,1],
    [-1,1,1],
    [1,-1,1]])

y = np.array([1,-1,1,-1,1,1])

eta=1
no_of_iteration=5
w=np.array([1,0,-1])
for j in range(no_of_iteration):
    print('Iteration',j)
    for i in range(X.shape[0]):

        wx=np.matmul(w.T,X[i])
        if wx>=0:
            sign=1
        else:
            sign=-1
        if sign!=y[i]:
            w=w+eta*y[i]*X[i]
        
        print(w)