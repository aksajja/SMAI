import numpy as np
import matplotlib.pyplot as plt

# Create a set of 100 samples in 2D in 2 classes
def generate_classes_1():
    # Different mean and same diagonal covariance
    class_A = []
    class_B = []

    diagonal_values = np.random.normal(size=2)
    covariance_matrix = np.array([[diagonal_values[0],0],[0,diagonal_values[0]]])
    
    u1 = np.random.normal(size=2)
    class_A = np.random.multivariate_normal(u1, covariance_matrix,100)

    u2 = np.random.normal(size=2)
    class_B =np.random.multivariate_normal(u2, covariance_matrix,100)

    print("mean1 = ", u1, " mean2 = ", u2)
    print("covariance matrix is ", covariance_matrix)
    print(class_A.shape, class_B.shape)
    return class_A, class_B


def sample_train_test(class_A, class_B):
    train_A = np.empty([0,2])
    train_B = np.empty([0,2]) 

    test_A =[] 
    test_B =[] 
    
    while len(train_A)<80:
        iter_index = np.random.randint(0,len(class_A))
        train_A = np.append(train_A,class_A[iter_index].reshape(1,2)).reshape(-1,2)
        class_A = np.delete(class_A,iter_index,axis=0)

    
    while len(train_B)<80:
        iter_index = np.random.randint(0,len(class_B))
        train_B = np.append(train_B, class_B[iter_index].reshape(1,2)).reshape(-1,2)
        class_B = np.delete(class_B,iter_index,axis=0)

    test_A = class_A
    test_B = class_B

    return train_A, train_B, test_A, test_B



'''
    X - input
    t1 - output of Layer 1
    Y - output of Layer 2
    L - output of Layer 3
    
    w1 - weights of Layer 1
    w2 - weights of Layer 2
'''

def get_MSE_errors(X, train_label):
    MSE_Error=[]
    weights_d = []
    d = 2
    hidden_neurons = 2
    output_neurons = 1
    w1=np.zeros((hidden_neurons,d))
    w2=np.zeros((output_neurons,d))
    Eta= 0.0001
    ErrorObs=[]
    for epoch in range(100):
        t1 = np.zeros((X.shape[0],hidden_neurons))
        Y = np.zeros((X.shape[0],output_neurons))
        
        # Feed-forward
        t1 = np.tanh(np.dot(w1,X.T),t1)
        Y = np.tanh(np.dot(w2.T,t1),Y)
        
        # Derivatives & Updates
        tan_derivative_2 = 1 - Y*Y
        # print(Y.shape, train_label.shape, tan_derivative_2.shape, t1.shape, X.shape)
        intermediate_value = np.dot((train_label-Y),tan_derivative_2)
        Delj2 = 2*np.sum((intermediate_value)*t1,axis=0).reshape(w2.shape)
        print(epoch, " : ", Y, Y*Y, intermediate_value.shape, Delj2)
        w2_new = w2 + Eta*Delj2
        w2_new = w2_new/np.max(w2_new)

        tan_derivative_1 = 1 - t1*t1
        # print(np.append(intermediate_value,intermediate_value,axis=1).shape)
        # int_value_1 = np.append(intermediate_value,intermediate_value,axis=1)*tan_derivative_1*X[:,0].reshape(-1,1)
        # int_value_2 = np.append(intermediate_value,intermediate_value,axis=1)*tan_derivative_1*X[:,1].reshape(-1,1)
        Delj1 = 2*np.sum(np.append(int_value_1,int_value_2,axis=1),axis=0).reshape(w1.shape)
        w1_new = w1 + Eta*Delj1
        w1_new = w1_new/np.max(w1_new,axis=0)

        # print(w1_new.shape, w2_new.shape, np.append(int_value_1,int_value_2,axis=1).shape)
        t1 = np.tanh(np.dot(X,w1_new.T),t1)
        Y = np.tanh(np.dot(t1,w2_new.T),Y)
        print(epoch, " : ", w1_new, w2_new)
        Err= np.sum((train_label-Y)**2,axis=0)
        ErrorObs.append(Err)
        w1 = w1_new
        w2 = w2_new
        break

    epochs = [iter for iter in range(0,100)]
    print(ErrorObs)
    plt.plot(epochs,ErrorObs)
    plt.title("Check 1")
    plt.show()
    # w_optimal =w
    # weights_d.append(w_optimal)
    # e = dataWithW(w_optimal, X[:d], d)
    # MSE_Error.append(e)

class_A, class_B = generate_classes_1()
train_A, train_B, test_A, test_B = sample_train_test(class_A,class_B)

train_A = np.asarray(train_A)
print(train_A.shape, train_A[1])
train_label_A = np.zeros([80,1])
train_label_B = np.ones([80,1])
get_MSE_errors(np.append(train_A,train_B).reshape(2,-1), np.append(train_label_A,train_label_B).reshape(1,-1))