import numpy as np

def generate_samples(n,d):
    np.random.seed(123)
    uniform_points = np.random.uniform(-5,5,size=(n,d))
    w = np.random.uniform(0,10,size=(1,d))
    print("fn mean : ", np.mean(w.dot(uniform_points.T)))
    x_mean = np.mean(uniform_points,axis=0).reshape(1,10)
    print("a^T*X mean : ",w.dot(x_mean.T))
    # uniform_points = uniform_points.reshape(1000)
    # uniform_points = np.sort(uniform_points)
    # lin_fuc = (3*uniform_points)+2
    # noise = np.random.normal(0, 0.1, 1000)
    # dataset = lin_fuc + noise
    # return (uniform_points, dataset)

d = 10
n = 100
generate_samples(n,d)