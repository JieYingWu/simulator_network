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
ensemble_size = 0
pc_length = 27000


def play_simulation(net, mesh, robot_pos, folder_name):
    steps = robot_pos.size()[0]

    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
    
    ## Run ##
    for i in range(steps):        
        cur_pos = robot_pos[i,1:utils.FIELDS+1].unsqueeze(0)
        correction = net(mesh, cur_pos)
        for j in range(ensemble_size):
            update = net(mesh, cur_pos)
            correction = correction + update
        network_correction = (correction / (ensemble_size+1)).detach()
        mesh = utils.correct(mesh, network_correction)
        write_out = mesh.clone().cpu().numpy().reshape(3,-1).transpose()
        np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)

if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt('../../dataset/2019-10-09-reduced/dvrk/' + folder_name  + '_robot_cartesian_velocity.csv', delimiter=',')
    robot_pos = torch.from_numpy(robot_pos).float().to(device)

    if len(sys.argv) == 2:
        network_path = Path('../deformable/augmentation_model.pt')
    elif len(sys.argv) == 3:
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
