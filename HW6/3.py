import math,random
import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols
from sympy.plotting import plot

Class_w1=np.array([(0,0),(0,1),(2,0),(3,2),(3,3),(2,2),(2,0)])
Class_w2=np.array([(7,7),(8,6),(9,7),(8,10),(7,10),(8,9),(7,11)])
data_set = np.concatenate((Class_w1, Class_w2), axis=0)

Class_w1_mean = np.mean(Class_w1,axis=0)
Class_w2_mean = np.mean(Class_w2,axis=0)
Cov_Class_w1=np.cov(Class_w1.T)
Cov_Class_w2=np.cov(Class_w2.T)

print("Mean of class w1",Class_w1_mean)
print("Mean of class w2",Class_w2_mean)
print("Cov class A",Cov_Class_w1)
print("Cov Class B",Cov_Class_w2)


def MultiVariate_MLE(X,class_mean,sigma):
	diff = X-class_mean
	p=sigma.shape[0]
	Sigma_inv=np.linalg.pinv(sigma)
	DetD=np.linalg.det(sigma)
	mp = ((-p/2)*np.log(2*(np.pi))) - (np.log(DetD)/2) - ((np.matmul(np.matmul(diff.T, Sigma_inv), diff))/2)
	return mp

def line_points(point, Class_A_mean, Class_B_mean, Cov_Class_w1, Cov_ClassB):
	g1 = MultiVariate_MLE(point,Class_A_mean,Cov_Class_w1)
	g2= MultiVariate_MLE(point,Class_B_mean,Cov_ClassB)
	gval=g1-g2
	return gval

def line_points_with_cost(point, Class_A_mean, Class_B_mean, Cov_Class_w1, Cov_ClassB, cost):
	g1 = MultiVariate_MLE(point,Class_A_mean,Cov_Class_w1)+2*cost
	g2= MultiVariate_MLE(point,Class_B_mean,Cov_ClassB)+cost
	gval=g1-g2
	return gval

x_points = np.arange(-1,11)
y_points=[]
y_points_cost=[]
for point in x_points:
	point = np.array([point, point])
	y= line_points(point, Class_w1_mean, Class_w2_mean, Cov_Class_w1, Cov_Class_w2)
	y_points.append(y)
	ynew = line_points_with_cost(point, Class_w1_mean, Class_w2_mean, Cov_Class_w1, Cov_Class_w2, cost=8)
	y_points_cost.append(ynew)
	
#plt.plot(x_points,y_points, color='green' , label='initial_decision_boundary')
plt.plot(x_points,y_points_cost, color='blue',  label='decision_boundary_with_cost')
plt.scatter(Class_w1[:,0],Class_w1[:,1], c="gray" )
plt.scatter(Class_w2[:,0],Class_w2[:,1], c="black")
plt.ylim(-20,20)
plt.legend(loc='upper left')
plt.show()
