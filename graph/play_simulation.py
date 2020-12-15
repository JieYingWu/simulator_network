import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import utils
from model import GraphUNet

import gc

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 

device = torch.device('cuda') 
ensemble_size = 0
pc_length = 27000


def closest_node(node, nodes):
    dist_2 = torch.sum((nodes - node)**2, axis=1)
    return torch.argmin(dist_2)

def play_simulation(net, mesh, robot_pos, folder_name):
    steps = robot_pos.size()[0]
    edge_index = utils.make_edges().to(device)
    
    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()

    ## Run ##
    for i in range(steps):        
        write_out = mesh.clone().cpu().numpy()
        np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)

        kinematics_arr = torch.zeros(mesh.size()[0], 6).float().to(device)
        cur_pos = robot_pos[i,:].unsqueeze(0)
        closest_point = closest_node(cur_pos[:,0:3], mesh)
        kinematics_arr[closest_point, :] = cur_pos

        correction = net(torch.cat((mesh, kinematics_arr), axis=1), edge_index)
        network_correction = correction.detach()
        mesh = mesh + utils.correct(mesh, network_correction)

if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    base_dir = '../../dvrk_soft_tissue_simulator/dataset/2019-10-09-GelPhantom1/'
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt(base_dir + 'dvrk/' + folder_name  + '_robot_cartesian_velocity.csv', delimiter=',')
    robot_pos = np.concatenate((robot_pos[:,1:4], robot_pos[:,8:11]), axis=1)
    robot_pos = torch.from_numpy(robot_pos).float().to(device)
    if len(sys.argv) == 2:
        network_path = Path('augmentation_model.pt')
    elif len(sys.argv) == 3:
        network_path = Path('checkpoints/models_diff/model_' + sys.argv[2] + '.pt')
    else:
        print('Too many arguments')
        exit()
    
    net = GraphUNet(9, 64, 3, 3).to(device)
    # Load previous model if requested
    if network_path.exists():
        state = torch.load(str(network_path))
        net.load_state_dict(state['model'])
        net = net.to(device)
        print('Restored model')
    else:
        print('Failed to restore model')
        print(network_path)
        net = net.to(device)

    base_mesh = 'mesh_fine.txt'
    mesh = torch.from_numpy(np.genfromtxt(base_mesh)).float().to(device)

    play_simulation(net, mesh, robot_pos, folder_name)
