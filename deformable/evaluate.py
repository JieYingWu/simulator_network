import os
import sys
import torch
import plyfile
import numpy as np
import utils
from chamferdist.chamferdist import ChamferDistance

mesh_path = sys.argv[1]
gt_path = sys.argv[2]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mesh_files = os.listdir(mesh_path)
mesh_files = sorted(mesh_files)
gt_files = sorted(os.listdir(gt_path))
loss = 0
loss_arr = np.zeros(len(mesh_files))
loss_fn = ChamferDistance()

#print('Using base mesh')
print('Evaluating ', len(mesh_files), ' files')
for i in range(len(mesh_files)):
    try:
        mesh = np.genfromtxt(mesh_path + mesh_files[i])
#        mesh = np.genfromtxt(mesh_path + mesh_files[0])
        mesh = torch.from_numpy(mesh)
    except:
        print("Can't find ", mesh_files[i])
        exit()
    mesh = mesh.reshape(utils.VOL_SIZE)#.permute(3,0,1,2)
#    print(mesh[:,-1,:,1])
    mesh = mesh[:,-1,:,:].unsqueeze(0).to(device).float()
#    mesh = utils.refine_mesh(mesh.permute((0,3,1,2)), 3, device).permute((0,2,3,1)) 
    mesh = mesh.reshape(-1,3).unsqueeze(0)

    try:
        pc = plyfile.PlyData.read(gt_path + gt_files[i+1])['vertex']
    except:
        print(i, ' is out of range')
        exit()
    pc = torch.from_numpy(np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)).unsqueeze(0).float().to(device)
    dist1, dist2, idx1, idx2 = loss_fn(mesh.contiguous(), pc.contiguous())
#    print(i,dist2.mean())
    dist2 = torch.sqrt(dist2)
    loss += torch.mean(dist2) #dist[0]
    loss_arr[i] = torch.mean(dist2)
    
print(loss/len(mesh_files))
np.savetxt('loss.csv', loss_arr, delimiter=',')
