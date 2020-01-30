import os
import sys
import torch
import plyfile
import numpy as np
from utils import refine_mesh
from chamferdist.chamferdist import ChamferDistance

mesh_path = sys.argv[1]
gt_path = sys.argv[2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mesh_files = os.listdir(mesh_path)
mesh_files = sorted(mesh_files)
gt_files = sorted(os.listdir(gt_path))
loss = 0
loss_fn = ChamferDistance()

for i in range(len(mesh_files)):
    try:
        mesh = np.genfromtxt(mesh_path + mesh_files[i])
#        mesh = np.genfromtxt(mesh_path + mesh_files[0])
        mesh = torch.from_numpy(mesh)
    except:
        print("Can't find ", mesh_files[i])
        exit()
    mesh = mesh.reshape(13, 5, 5, 3)
#    print(mesh[:,-1,:,1])
    mesh = mesh[:,-1,:,:]
#    mesh = refine_mesh(mesh.unsqueeze(0), 3, device)
 
    mesh = mesh.reshape(-1,3).unsqueeze(0).float().to(device)

    try:
        pc = plyfile.PlyData.read(gt_path + gt_files[i])['vertex']
    except:
        print(i, ' is out of range')
        exit()
    pc = torch.from_numpy(np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)).unsqueeze(0).float().to(device)
    dist1, dist2, idx1, idx2 = loss_fn(mesh.contiguous(), pc.contiguous())
#    print(dist2.mean())
    loss += torch.mean(dist2) #dist[0]

print(loss/len(mesh_files))
