import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

batch = 23

epoch = sys.argv[1]
mesh = np.load('checkpoints/results/prediction_'+str(epoch)+'.npy')

mesh_subset = mesh[batch,:,:,:,:].reshape(3,-1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(mesh.shape)
x = mesh_subset[0,:]
y = mesh_subset[1,:]
z = mesh_subset[2,:]

ax.scatter3D(x, y, z, c='gray', marker='o', label='simulation')
plt.show()
