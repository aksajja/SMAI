import  pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import math
from itertools import combinations

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1
 
pd_tr = pd.DataFrame()
tr_y = pd.DataFrame()
 
for i in range(1,6):
    data = unpickle('./cifar-10-batches-py/data_batch_' + str(i))
    pd_tr = pd_tr.append(pd.DataFrame(data[b'data']))
    tr_y = tr_y.append(pd.DataFrame(data[b'labels']))
    pd_tr['labels'] = tr_y
 
tr_x = np.asarray(pd_tr.iloc[:, :3072])
tr_y = np.asarray(pd_tr['labels']) 
labels = unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']

principal_eigen_vectors={}
ErrorDistances={}
def Mean_vector(train_data,class_size,feature_size):
    mean_matrix=np.zeros((class_size, feature_size))
    for i in range (10):
        print(i)
        labels=np.where(tr_y==i)
        sample_data_label=train_data[labels,:]
        sample_data_label.shape=(np.size(labels,1),feature_size)
        cov_data=np.cov(sample_data_label.T)
        Eigen_valuei , Eigen_vectori = LA.eig(cov_data)
        Normalized_Eigen_valuei=Eigen_valuei/np.amax(Eigen_valuei)
        principal_eigen_vector_20=Eigen_vectori[:,0:20]
        principal_eigen_vector_20=principal_eigen_vector_20.T
        projected_data=np.matmul(principal_eigen_vector_20,sample_data_label.T)
        principal_eigen_vectors[i]=principal_eigen_vector_20
        inv_eigen_vec=np.linalg.pinv(principal_eigen_vector_20)
        reconstrucion_image=np.matmul(inv_eigen_vec, projected_data)
        reconstrucion_image=reconstrucion_image.T
        distance=(1/(5000*3072))*np.sum((reconstrucion_image-sample_data_label)**2)
        ErrorDistances[i]=distance
        mean_matrix[i,:]=np.mean(sample_data_label,axis=0)
        x=np.mean(sample_data_label,axis=0)
    return ErrorDistances,mean_matrix

ErrorDistances,mean1= Mean_vector(tr_x,10,3072)

n=list(range(0,10))
errors=ErrorDistances.values()
plt.show()
plt.plot(n,errors,marker='*')
plt.xlabel('Category')
plt.ylabel('Error in reconstruction')
plt.title('plot for a bit')

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
Arr=[0,1,2,3,4,5,6,7,8,9]
Mean_distances={}
classes = list(combinations(Arr, 2))
for n in range (0,45) :
        [i,j]=classes[n]
        dist=euclideanDistance(mean1[i,:],mean1[j,:])
        Mean_distances[classes[n]]=dist
print("Mean distances",Mean_distances)

for i in range(10):
    labels=np.where(tr_y==i)
    sample_data_label=tr_x[labels,:]
    sample_data_label.shape=(np.size(labels,1),3072)
    print(sample_data_label.shape)
    meanx=mean1[i,:]
    meanx.shape=(1,3072)
    mean_repeat=np.repeat(meanx,5000,axis=0)
    subtracted_matrix=np.subtract(sample_data_label,mean_repeat)
    Error_List=[]
    for j in range(10):
        if(i!=j):
            prj_data=np.matmul(principal_eigen_vectors[j],sample_data_label.T)
            inv_vector=np.linalg.pinv(principal_eigen_vectors[j])
            Recon_image=np.transpose(np.matmul(inv_vector,prj_data))
            Error=(1/(5000*3072))*np.sum((Recon_image-subtracted_matrix)**2)
        elif(i==j):
            Error=0
        Error_List.append(Error)
        sorted_indexes=np.argsort(np.asarray(Error_List))
    print("For Class",i , sorted_indexes[1:4])
print(principal_eigen_vectors[0].shape)
