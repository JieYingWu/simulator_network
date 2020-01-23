import os
import sys
import torch
import plyfile
import numpy as np
from chamferdist.chamferdist import ChamferDistance

def refine_mesh(mesh, factor):
    x,y,z =  mesh.size()
    new_x = (x-1)*factor+1
    new_y = (y-1)*factor+1
    fine_mesh = torch.zeros([new_x, new_y, z])

    # Fill out the rows 
    for i in range(x):
        cur_row = mesh[i,:,:]
        for j in range(y-1):
            cur_elem = cur_row[j,:]
            next_elem = cur_row[j+1,:]
            for k in range(factor):
                weight = float(k)
                fine_mesh[i*factor,j*factor+k,:] = (factor-weight)/factor*cur_elem + weight/factor*next_elem
        fine_mesh[i*factor,-1,:] = mesh[i,-1,:]
                
    # Fill out the columns
    for j in range(new_y):
        cur_col = fine_mesh[:,j,:]
        for i in range(x-1):
            cur_elem = cur_col[i*factor,:]
            next_elem = cur_col[(i+1)*factor,:]
            for k in range(factor):
                weight = float(k)
                fine_mesh[i*factor+k,j,:] = (factor-weight)/factor*cur_elem + weight/factor*next_elem

    return fine_mesh


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
        mesh = torch.from_numpy(mesh)
    except:
        print("Can't find ", mesh_files[i])
        exit()
    mesh = mesh.reshape(13, 5, 5, 3)
#    print(mesh[:,-1,:,1])
    mesh = mesh[:,-1,:,:]
    mesh = refine_mesh(mesh, 3)
 
    mesh = mesh.reshape(-1,3).unsqueeze(0).float().to(device)

    try:
        pc = plyfile.PlyData.read(gt_path + gt_files[i])['vertex']
    except:
        print(i, ' is out of range')
        exit()
    pc = torch.from_numpy(np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)).unsqueeze(0).float().to(device)
    dist1, dist2, idx1, idx2 = loss_fn(mesh.contiguous(), pc.contiguous())
    print(dist2.mean())
    loss += torch.mean(dist2) #dist[0]

print(loss/len(mesh_files))
