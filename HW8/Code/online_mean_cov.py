import numpy as np
import math

x = np.random.uniform(1,10,10)
y = np.random.uniform(1,10,10)
x = x.reshape([x.shape[0],1])
y = y.reshape([y.shape[0],1])

# The slope of the linear regression, in fact, is the covariance divided by the variance of the independent variable, sx^2. Passing through the mean.
def compute_linear_regression(cov, var_x, mean_x, mean_y):
    slope = cov/var_x
    y_intercept = mean_y-(slope*mean_x)
    return slope, y_intercept

def iterative_mean_cov(x,y):
    # Direct computation
    matrix_xy = np.append(x,y,axis=1)
    print("\nDirectly computed --- ")
    print("Mean (x,y) ", np.mean(x,axis=0), np.mean(y,axis=0))
    print("Cov  ", np.cov(matrix_xy.T))
    print("\n\n ---- \n\n")

    # Read values 1 at a time and compute mean/cov
    mean_x = x[0][0]
    mean_y = y[0][0]
    cov = 0
    std_dev_x = 0
    std_dev_y = 0
    for i in range(1,x.shape[0]):
        old_mean_x = mean_x
        old_mean_y = mean_y
        mean_x = mean_x + (x[i][0]-mean_x)/(i+1)
        mean_y = mean_y + (y[i][0]-mean_y)/(i+1)

        std_dev_x = std_dev_x + (x[i][0]-old_mean_x)*(x[i][0]-mean_x)
        std_dev_y = std_dev_y + (x[i][0]-old_mean_x)*(x[i][0]-mean_x)

        var_x = std_dev_x/i
        var_y = std_dev_y/i

        old_cov = cov
        cov = ((i-1)/i)*old_cov + ((i/pow(i+1,2))*(x[i][0]-old_mean_x)*(y[i][0]-old_mean_y) + (x[i][0] - mean_x)*(y[i][0] - mean_y))/i
        print("Iteration ", i, " ---")
        print("Mean(x,y): ",mean_x, mean_y)
        print("cov(x,y): ",cov)
        print("Regression (slope, intercept) ", compute_linear_regression(cov, var_x, mean_x, mean_y))

    print("\nFinal values --- ")
    print("Mean (x,y) ", mean_x, mean_y)
    print("Cov  ", cov)

# Running window.
def running_window(x,y,window_size=3):
    for i in range(x.shape[0]-3):
        window_x = np.split(x,(i,i+3),axis=0)[1]
        window_y = np.split(y,(i,i+3),axis=0)[1]
        matrix_xy = np.append(window_x,window_y,axis=1)
        mean_x = np.mean(window_x)
        mean_y = np.mean(window_y)
        cov = np.cov(matrix_xy.T)
        var_x = np.var(window_x)
        print("Iteration ", i+1, "-- ")
        print("Mean_x: ",mean_x, " Mean_y: ",mean_y)
        print("Cov: ", cov)
        print("Regression (slope, intercept) ", compute_linear_regression(cov, var_x, mean_x, mean_y))

iterative_mean_cov(x,y)
running_window(x,y,3)

