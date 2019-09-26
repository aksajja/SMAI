import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import math

pts = np.array([[0,0],[5,0],[7,2],[4,7],[2,2]])
print(pts[0], pts.shape)
pts_x = [0,5,7,2,4,0]
pts_y = [0,0,2,2,7,0]
pts = np.asarray(pts)
mean = np.mean(pts,axis=0)
covariance = np.cov(pts.T)
print("Mean : ", mean)
print("Covariance : ", covariance)

eigen_vectors, eigen_values, _ = np.linalg.svd(covariance)

print(eigen_vectors)
print(eigen_values)

fig = plt.figure()
plots1 = fig.add_subplot(121)
plots1.plot(pts_x,pts_y)
plots1.plot([0,eigen_vectors[0][0]], [0,eigen_vectors[1][0]])
plots1.plot([0,eigen_vectors[0][1]], [0,eigen_vectors[1][1]])

transformed_pts = []
for i in range(pts.shape[0]):
    new = eigen_vectors.dot(pts[i].T)
    transformed_pts.append(new)
transformed_pts = np.asarray(transformed_pts)
print(transformed_pts)
print(transformed_pts[:,0], transformed_pts[:,1])

plots2 = fig.add_subplot(122)
plots2.scatter(transformed_pts[:,0], transformed_pts[:,1])
transformed_pts_x = np.append(transformed_pts[:,0], transformed_pts[0][0])
transformed_pts_y = np.append(transformed_pts[:,1], transformed_pts[0][1])
plots2.plot(transformed_pts_x,transformed_pts_y)
plt.show()

## Change in angle between first last adjacent sides of the polygon.
original_angle = 0
new_angle = 0

original_side_1 = np.asarray(pts[4]-pts[3]).reshape([1,2])
original_side_2 = np.asarray(pts[0]-pts[4]).reshape([1,2])
original_dot_product = original_side_1.dot(original_side_2.T)
original_product_norms = (np.linalg.norm(original_side_1)*np.linalg.norm(original_side_2))
original_angle = math.acos(original_dot_product/original_product_norms)
print("For original parallelogram ----")
print("Dot : ", original_dot_product)
print("Product of norms: ", original_product_norms)
print("Angle: ", original_angle)
transformed_side_1 = np.asarray(transformed_pts[4]-transformed_pts[3]).reshape([1,2])
transformed_side_2 = np.asarray(transformed_pts[0]-transformed_pts[4]).reshape([1,2])
transformed_dot_product = transformed_side_1.dot(transformed_side_2.T)
transformed_product_norms = (np.linalg.norm(transformed_side_1)*np.linalg.norm(transformed_side_2))
transformed_angle = math.acos(transformed_side_1.dot(transformed_side_2.T)/(np.linalg.norm(transformed_side_1)*np.linalg.norm(transformed_side_2)))
print("\n\nFor transformed parallelogram ----")
print("Dot : ", transformed_dot_product)
print("Product of norms: ", transformed_product_norms)
print("Angle: ", transformed_angle)

'''
    For a parallelogram mean is the centroid of it.
'''

