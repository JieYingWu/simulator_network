import torch
import numpy as np
from dataset import SimulatorDataset3D
from torch.utils.data import DataLoader
from chamfer_distance.chamfer_distance import ChamferDistance

path = '../../dataset/2019-09-07-GelPhantom1'
data = 'test'
kinematics_path = [path+'/dvrk/data0_robot_cartesian_processed_interpolated.csv']
simulator_path = [path+'/simulator/' + data + '/']
label_path = [path+'/camera/' + data + '/']

img_size = [3, 25, 9, 9]
dataset = SimulatorDataset3D(kinematics_path, simulator_path, label_path, img_size)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

loss_fn = ChamferDistance()

for i, (kinematics, mesh, label) in enumerate(loader):
    mesh_subset = mesh[:,:,:,-1,:].reshape(mesh.size()[0],3,-1)
    label = label[:,:,~(label==0).all(axis=1)[0]]
    mesh_subset = torch.transpose(mesh_subset, 1,2)
    label = torch.transpose(label, 1,2)
    mesh_subset = mesh_subset[:,-10:-1,:]
    label = label[:,0:5,:]
    print(mesh_subset)
    print(label)
    loss = loss_fn(mesh_subset, label)
    print(torch.mean(loss[0]) + torch.mean(loss[1]))
