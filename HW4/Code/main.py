import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import random, linalg
from sklearn.datasets import make_spd_matrix

def generate_classes_1():
    # Different mean and same diagonal covariance
    class_A = []
    class_B = []

    diagonal_values = np.random.normal(size=3)
    covariance_matrix = np.array([[diagonal_values[0],0,0],[0,diagonal_values[0],0],[0,0,diagonal_values[0]]])
    
    u1 = np.random.normal(size=3)
    class_A = np.random.multivariate_normal(u1, covariance_matrix,1000)

    u2 = np.random.normal(size=3)
    class_B =np.random.multivariate_normal(u2, covariance_matrix,1000)

    print(" For Case I, ")
    print("mean1 = ", u1, " mean2 = ", u2)
    print("covariance matrix is ", covariance_matrix)
    return class_A, class_B

def generate_classes_2():
    # Different mean and same spd covariance
    class_A = []
    class_B = []
    

    u1 = np.random.normal(size=3)
    matrixSize = 3 
    A = random.rand(matrixSize,matrixSize)
    # covariance_matrix = np.dot(A,A.transpose())
    covariance_matrix = make_spd_matrix(3)
    class_A = np.random.multivariate_normal(u1, covariance_matrix,1000)

    u2 = np.random.normal(size=3)
    class_B =np.random.multivariate_normal(u2, covariance_matrix,1000)

    print(" For Case II, ")
    print("mean1 = ", u1, " mean2 = ", u2)
    print("covariance matrix is ", covariance_matrix)
    return class_A, class_B

def generate_classes_3():
    # Same mean and different covariance
    class_A = []
    class_B = []

    u = np.random.normal(size=3)
    diagonal_values = np.random.normal(size=3)
    covariance_matrix_1 = np.array([[diagonal_values[0],0,0],[0,diagonal_values[0],0],[0,0,diagonal_values[0]]])
    class_A = np.random.multivariate_normal(u, covariance_matrix_1,1000)

    diagonal_values = np.random.normal(size=3)
    covariance_matrix_2 = np.array([[diagonal_values[0],0,0],[0,diagonal_values[0],0],[0,0,diagonal_values[0]]])
    class_B =np.random.multivariate_normal(u, covariance_matrix_2,1000)

    print(" For Case III, ")
    print("mean = ", u)
    print("covariance matrix 1 : ", covariance_matrix_1 )
    print("covariance matrix 2 : ", covariance_matrix_2 )
    return class_A, class_B


def plot_case(class_A, class_B):

    fig = plt.figure()
    plot3d = fig.add_subplot(221, projection= '3d')
    plot3d.scatter(class_A[:,0],class_A[:,1],class_A[:,2],marker="o")
    plot3d.scatter(class_B[:,0],class_B[:,1],class_B[:,2],marker='*')
    plot3d.set_xlabel('X Label')
    plot3d.set_ylabel('Y Label')
    
    plotxy = fig.add_subplot(222)
    plotyz = fig.add_subplot(223)
    plotzx = fig.add_subplot(224)

    plotxy.scatter(class_A[:,0],class_A[:,1],marker="o")
    plotxy.scatter(class_B[:,0],class_B[:,1],marker='*')
    plotyz.scatter(class_A[:,1],class_A[:,2],marker="o")
    plotyz.scatter(class_B[:,1],class_B[:,2],marker='*')
    plotzx.scatter(class_A[:,2],class_A[:,0],marker="o")
    plotzx.scatter(class_B[:,2],class_B[:,0],marker='*')

    plotxy.set_xlabel('X axis')
    plotxy.set_ylabel('Y axis')

    plotyz.set_xlabel('Y axis')
    plotyz.set_ylabel('Z axis')

    plotzx.set_xlabel('Z axis')
    plotzx.set_ylabel('X axis')

    plt.show()    

if __name__=="__main__":
    
    # 3 cases
    # case 1
    # class_A, class_B = generate_classes_1()
    # plot_case(class_A,class_B)
    # case 2
    class_A, class_B = generate_classes_2()
    plot_case(class_A,class_B)
    # case 3
    # class_A, class_B = generate_classes_3()
    # plot_case(class_A,class_B)
    