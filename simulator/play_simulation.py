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
from model import UNet3D, SimuNetWithSurface, SimuAttentionNet
import random
import plyfile

import gc

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 

device = torch.device('cuda') 
ensemble_size = 0
pc_length = 27000


def play_simulation(net, mesh, robot_pos, folder_name):
    steps = robot_pos.size()[0]

    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
    
    ## Run ##
    for i in range(steps):        
        write_out = mesh.clone().cpu().numpy().reshape(3,-1).transpose()
        np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)

        cur_pos = robot_pos[i,:].unsqueeze(0)
        correction = net(mesh, cur_pos)
        for j in range(ensemble_size):
            update = net(mesh, cur_pos)
            correction = correction + update
        network_correction = (correction / (ensemble_size+1)).detach()
        mesh = mesh + utils.correct(mesh, network_correction)


if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    base_dir = '../../dataset/2019-10-09-GelPhantom1/'
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt(base_dir + 'dvrk/' + folder_name  + '_robot_cartesian_velocity.csv', delimiter=',')
    robot_pos = np.concatenate((robot_pos[:,1:4], robot_pos[:,8:11]), axis=1)
    robot_pos = torch.from_numpy(robot_pos).float().to(device)
    if len(sys.argv) == 2:
        network_path = Path('../deformable/augmentation_model.pt')
    elif len(sys.argv) == 3:
        network_path = Path('../deformable/checkpoints/models/model_' + sys.argv[2] + '.pt')
    else:
        print('Too many arguments')
        exit()
    
    net = UNet3D(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS)
    # Load previous model if requested
    if network_path.exists():
        state = torch.load(str(network_path))
        net.load_state_dict(state['model'])
        net = net.to(device)
        print('Restored model')
    else:
        print('Failed to restore model')
        print(network_path)
#        exit()
        net = net.to(device)


    ## Load mesh ##
#    mesh_path = base_dir + '/simulator/' + folder_name + '/'
#    mesh_files = os.listdir(mesh_path)
#    mesh_files = sorted(mesh_files)
    base_mesh = 'mesh_fine.txt'#mesh_path + mesh_files[0]
    mesh = torch.from_numpy(utils.reshape_volume(np.genfromtxt(base_mesh))).float().unsqueeze(0).to(device)

    play_simulation(net, mesh, robot_pos, folder_name)
