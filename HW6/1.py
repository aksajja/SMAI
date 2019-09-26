import numpy as np
import math
import matplotlib.pyplot as plt


def generate_samples(n):
    return np.random.uniform(0,10,n)

def calculate_y(x,u,var):
    return math.sin(x)+np.random.normal(u,var)

def generate_K_folds(samples, k):
    n = len(samples)
    fold_limit = int(n/k)
    k_folds = np.zeros([k,fold_limit],dtype=np.float32)
    for sample in samples:
        index = np.random.randint(0,k)
        while len(np.where(k_folds[index]==0.0)[0])==0:
            index = (index+1)%k
            print(index)
        non_zero_index = np.where(k_folds[index]==0.0)[0][0]
        k_folds[index][non_zero_index] = sample
    
    return k_folds

# def get_y(x):
#     return m*x+c


# def find_error(x,y,m,c):
#     np.sum(np.apply_along_axis(get_y,0,x,m,c),axis=0)

#     return err

def find_error(x,y,m,c):
    error_sum = 0
    for i in range(len(x)):
        error_sum += (y[i]-x[i])**2
    return math.sqrt(error_sum)

def optimal_K_fold(k_folds):
    y_matrix = np.empty([0,k_folds.shape[1]])
    for _fold in k_folds:
        y = []
        for x in _fold:
            y.append(calculate_y(x,0.05,0.2))
        y_matrix = np.append(y_matrix, [y],axis=0)

    min_error = 999999
    for i in range(len(k_folds)):
        if i ==0 :
            training_set = np.append([],k_folds[0:-1])
            training_y = np.append([],y_matrix[0:-1])
        elif i == len(k_folds)-1 :
            training_set = np.append([],k_folds[1:])
            training_y = np.append([],y_matrix[1:])
        else:
            training_set = np.append(k_folds[0:i],k_folds[i+1:])
            training_y = np.append(y_matrix[0:i],y_matrix[i+1:])
        test_set = k_folds[i]
        test_y = y_matrix[i]
        m,c = np.polynomial.polynomial.polyfit(training_set,training_y,1)
        err1 = find_error(test_set,test_y,m,c)
        if err1<min_error:
            u = np.mean(training_y, axis=0)
            var = np.var(training_y, axis=0)

    # return np.asarray(u,var).reshape([1,2])
    return u,var
        


def plot_K_folds(k_vals, y_vals, sample_len):
    plt.plot(k_vals, y_vals)
    plt.show()


# Case 1
samples = generate_samples(100)
k_vals = [5,10,20,25,50,100]
# mean_var_arr = np.empty([0,2])
means = []
var_list = []
for k in k_vals:
    k_folds = generate_K_folds(samples, k)
    # mean_var_arr = np.append(mean_var_arr ,optimal_K_fold(k_folds),axis=0)
    u,var = optimal_K_fold(k_folds)
    means.append(u)
    var_list.append(var)
plot_K_folds(k_vals, means, len(samples))
plot_K_folds(k_vals, var_list, len(samples))

# # Case 2
samples = generate_samples(10000)
k_vals = [5,10,20,25,50,100]
# generate_K_folds()
means = []
var_list = []
for k in k_vals:
    k_folds = generate_K_folds(samples, k)
    # mean_var_arr = np.append(mean_var_arr ,optimal_K_fold(k_folds),axis=0)
    u,var = optimal_K_fold(k_folds)
    means.append(u)
    var_list.append(var)
plot_K_folds(k_vals, means, len(samples))
plot_K_folds(k_vals, var_list, len(samples))
