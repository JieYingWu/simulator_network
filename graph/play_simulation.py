import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import utils
from model import GraphUNet

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 

device = torch.device('cuda') 

def closest_node(node, nodes):
    dist2 = torch.sum((nodes - node)**2, axis=1)
    min_value = torch.min(dist2)
    if min_value > utils.MAX_DIST:
        min_index = None
    else:
        min_index = torch.argmin(dist2)
    return min_index

def play_simulation(net, mesh, robot_pos, folder_name):
    base_mesh = 'mesh_fine.txt'
    base_mesh = torch.from_numpy(np.genfromtxt(base_mesh)).float().to(device)
    
    steps = robot_pos.size()[0]
    edge_index = utils.make_edges().to(device)
    start_time = time.perf_counter()
    ## Run ##
    for i in range(steps):        
        write_out = mesh.clone().cpu().numpy()
        np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)

        kinematics_arr = torch.zeros(mesh.size()[0], 6).float().to(device)
        cur_pos = robot_pos[i,:].unsqueeze(0)
        closest_point = closest_node(cur_pos[:,0:3], mesh)
        if closest_point is not None:
            cur_pos[:,0:3] = cur_pos[:,0:3] - mesh[closest_point,:]
            kinematics_arr[closest_point, :] = cur_pos

        correction = net(torch.cat((mesh-base_mesh, kinematics_arr), axis=1), edge_index)
        network_correction = utils.correct(mesh, correction.detach(), device)
#        print(torch.mean(network_correction, axis=0))
        mesh = mesh + network_correction

    end_time = time.perf_counter()
    print(start_time, end_time)
    print('Simulation took ' + str(end_time-start_time) + 's')
if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    base_dir = '../../dataset/2019-10-09-GelPhantom1/'
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt(base_dir + 'dvrk/' + folder_name  + '_robot_cartesian_velocity.csv', delimiter=',')
    robot_pos = np.concatenate((robot_pos[:,1:4], robot_pos[:,8:11]), axis=1)
    robot_pos = torch.from_numpy(robot_pos).float().to(device)
#    robot_pos = robot_pos[::10,:]
    if len(sys.argv) == 2:
        network_path = Path('augmentation_model.pt')
    elif len(sys.argv) == 3:
#        network_path = Path('checkpoints/UNet/fine_mesh_no_aug/model_' + sys.argv[2] + '.pt')
        network_path = Path('checkpoints/UNet/fine_mesh_half_step/model_' + sys.argv[2] + '.pt')
    else:
        print('Too many arguments')
        exit()
    
    net = GraphUNet(utils.IN_CHANNELS, 256, 3, utils.NET_DEPTH, sum_res=False).to(device)
    # Load previous model if requested
    if network_path.exists():
        state = torch.load(str(network_path), map_location=device)
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
