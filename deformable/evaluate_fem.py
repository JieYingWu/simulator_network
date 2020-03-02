import os
import sys
import torch
import plyfile
import numpy as np
import utils
import torch.nn as nn
from chamferdist.chamferdist import ChamferDistance

mesh_path = sys.argv[1]
gt_path = sys.argv[2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mesh_files = os.listdir(mesh_path)
mesh_files = sorted(mesh_files)
gt_files = sorted(os.listdir(gt_path))
loss = 0
loss_fn = nn.MSELoss()

#print('Using base mesh')
for i in range(len(mesh_files)):
    try:
        mesh = np.genfromtxt(mesh_path + mesh_files[i])
#        mesh = np.genfromtxt(mesh_path + mesh_files[0])
        mesh = torch.from_numpy(mesh)
    except:
        print("Can't find ", mesh_files[i])
        exit()
    mesh = mesh.reshape(utils.VOL_SIZE)#.permute(3,0,1,2)
#    mesh = refine_mesh(mesh.unsqueeze(0), 3, device)
 
    try:
        fem = np.genfromtxt(gt_path + gt_files[i])
#        mesh = np.genfromtxt(mesh_path + mesh_files[0])
        fem = torch.from_numpy(fem)
    except:
        print("Can't find ", gt_files[i])
        exit()
    fem = fem.reshape(utils.VOL_SIZE)#.permute(3,0,1,2)

    loss += loss_fn(mesh, fem)
    
print(loss/len(mesh_files))
