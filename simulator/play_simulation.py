import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
sys.path.insert(0,'../processing/')
sys.path.insert(0,'../deformable/')
import utils
import geometry_util as geo
from model import UNet3D, SimuNet, SimuNetWithSurface, SimuAttentionNet
import random
import plyfile

import gc

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 

device = torch.device('cuda') 
ensemble_size = 19
pc_length = 27000
#for m in net.modules():
#  if isinstance(m, nn.BatchNorm3d):
#    m.eval()


def play_simulation(net, mesh, robot_pos, folder_name):
    steps = robot_pos.size()[0]

    path = '../../dataset/2019-10-09-reduced/camera/' + folder_name + '_filtered/' 
    files = sorted(os.listdir(path))

    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
    
    ## Run ##
    for i in range(steps):
        # Next point cloud
#        pc = plyfile.PlyData.read(path+files[i])['vertex']
#        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
#        indices = range(pc.shape[0])
#        random.shuffle(indices)
#        pc = pc[indices[0:pc_length], :]
#        pc = torch.from_numpy(np.transpose(pc, (1,0))).float().to(device).unsqueeze(0)
        
        cur_pos = robot_pos[i,0:1+utils.FIELDS].unsqueeze(0)
#        mesh_kinematics = utils.concat_mesh_kinematics(mesh, cur_pos)
#        print(mesh_kinematics[0,:,1,1,1])
#        print(mesh_kinematics[0,:,2,2,2])
#        print(mesh_kinematics[0,:,1,-1,1])
#        print(mesh_kinematics[0,:,0,-1,1])        
#        print(mesh_kinematics[0,:,2,-1,1])
#        exit()
        correction = net(mesh, cur_pos)
        for j in range(ensemble_size):
            update = net(mesh, cur_pos)
#        print(update[0,1,:,-1,:])
            correction = correction + update
#    exit()
        network_correction = (correction / (ensemble_size+1)).detach()
#    print(network_correction[0,0,0])
#    print(mesh[0,2,0,4,4])
        mesh = utils.correct(mesh, network_correction)
#    print(mesh[0,1,:,-1,:])
        write_out = mesh.clone().cpu().numpy().reshape(3,-1).transpose()
        np.savetxt(folder_name + "/position" + '%04d' % (i+1) + ".txt", write_out)

if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt('../../dataset/2019-10-09-reduced/dvrk/' + folder_name  + '.csv', delimiter=',')
    robot_pos = torch.from_numpy(robot_pos).float().to(device)

    if len(sys.argv) == 3:
        network_path = Path('../deformable/checkpoints/models/model_' + sys.argv[2] + '.pt')
    else:
        print('Too many arguments')
        exit()
    
    net = SimuAttentionNet(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS, dropout=utils.DROPOUT)
    # Load previous model if requested
    if network_path.exists():
        state = torch.load(str(network_path))
        net.load_state_dict(state['model'])
        net = net.to(device)
        print('Restored model')
    else:
        print('Failed to restore model')
#        exit()
        net = net.to(device)


    ## Load mesh ##
    simulator_file = 'mesh.txt'
    mesh = torch.from_numpy(utils.reshape_volume(np.genfromtxt(simulator_file))).float().unsqueeze(0).to(device)

    play_simulation(net, mesh, robot_pos, folder_name)
