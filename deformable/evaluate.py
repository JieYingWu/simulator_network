import os
import sys
import torch
import plyfile
import numpy as np
from chamfer_distance.chamfer_distance import ChamferDistance
#from scipy.spatial.distance import directed_hausdorff

old_naming = False
mesh_path = sys.argv[1]
gt_path = sys.argv[2]

mesh_files = os.listdir(mesh_path)
if (old_naming):
    mesh_order = np.array([int(x.split('.')[0][8:]) for x in mesh_files])
    mesh_order = np.argsort(mesh_order).astype(int)
    mesh_files = [mesh_files[i] for i in mesh_order]
else:
    mesh_files = sorted(mesh_files)
print(mesh_files[0:20])
gt_files = sorted(os.listdir(gt_path))
loss = 0
loss_fn = ChamferDistance()

for i in range(len(mesh_files)):
    try:
        mesh = np.genfromtxt(mesh_path + mesh_files[i])
        mesh = torch.from_numpy(mesh)
    except:
        print("Can't find ", mesh_files[i])
        exit()
    mesh = mesh.reshape(25, 9, 9, 3)
    mesh = mesh[:,-1,:,:].reshape(-1,3).unsqueeze(0).float()

    try:
        pc = plyfile.PlyData.read(gt_path + gt_files[i])['vertex']
    except:
        print(i, ' is out of range')
        exit()
    pc = torch.from_numpy(np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)).unsqueeze(0).float()
#    dist = directed_hausdorff(pc, mesh)
    dist1, dist2 = loss_fn(pc, mesh)
    loss += torch.mean(dist1) #dist[0]

print(loss/len(mesh_files))
