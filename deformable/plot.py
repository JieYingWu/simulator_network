import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


scale = np.zeros((1,3,1,1))
scale[0,:,0,0] = np.array([5.28, 7.16, 7.86])/2

def correct(mesh,x):
    corrected = mesh
    x = (x-0.5)*scale
    corrected = mesh + x
    return corrected

batch = 29

epoch = sys.argv[1]
mesh = np.load('checkpoints/results/mesh_'+str(epoch)+'.npy')
label = np.load('checkpoints/results/label_'+str(epoch)+'.npy')
pred = np.load('checkpoints/results/prediction_'+str(epoch)+'.npy')
pred = correct(mesh, pred)
label = label[batch]

label = label[:,~(label==0).all(axis=0)]
#mesh_subset = mesh[batch,:,-1,:].reshape(3,-1)
#pred_subset = pred[batch,:,-1,:].reshape(3,-1)
mesh_subset = mesh[batch,:,:,:].reshape(3,-1)
pred_subset = pred[batch,:,:,:].reshape(3,-1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(mesh.shape)
x = mesh_subset[0,:]
y = mesh_subset[1,:]
z = mesh_subset[2,:]

ax.scatter3D(x, y, z, c='black', marker='o', label='simulation')

x = label[0,::50]
y = label[1,::50]
z = label[2,::50]

ax.scatter3D(x, y, z, c='blue', marker='o', label='gt')

x = pred_subset[0,:]
y = pred_subset[1,:]
z = pred_subset[2,:]

ax.scatter3D(x, y, z, c='green', marker='o', label='prediction')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()

plt.show()
