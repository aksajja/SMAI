import numpy as np
import random
# part(a)
import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(0)

def calc_variance_constantS(K_values, S):
    Variance_est=np.zeros(np.size(K_values,0))
    for i in range (np.size(K_values,0)):
        K=K_values[i]
        sets=np.int(K/S) 
        X=np.random.standard_normal(size=(S,sets)) 
        mean=np.mean(X,1) 
        Variance_est[i]=np.var(mean)
    print(Variance_est)
    return Variance_est

def calc_variance_constantK(total_samples, Sets):
    Variance_est1=np.zeros(np.size(Sets,0))
    for i in range(len(Sets)):
        S=Sets[i]
        sets=np.int(total_samples/S) 
        X=np.random.standard_normal(size=(S,sets)) 
        mean=np.mean(X,1)
        Variance_est1[i]=np.var(mean)
    return Variance_est1

K_values=[50,100,250,500,1000]
S=10
Variance_1 = calc_variance_constantS(K_values, S)
plt.plot(K_values,Variance_1)
plt.xlabel('No. of Samples')
plt.ylabel('Variance')
plt.title('Variance Vs K with constant S')
plt.show()

total_samples=10000
Sets=[2,10,50,200,1000]
Variance_2 = calc_variance_constantK(total_samples, Sets)
plt.plot(Sets,Variance_2)
plt.xlabel('No. of samples')
plt.ylabel('Variance')
plt.title('Variance Vs S with constant K')
plt.show()