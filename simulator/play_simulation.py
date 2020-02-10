import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
sys.path.insert(0,'../processing/')
sys.path.insert(0,'../deformable/')
import utils
import geometry_util as geo
from model import UNet3D, SimuNet

import gc

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 

device = torch.device('cuda') 
ensemble_size = 20

#for m in net.modules():
#  if isinstance(m, nn.BatchNorm3d):
#    m.eval()


def play_simulation(net, mesh, robot_pos, folder_name):
    steps = robot_pos.size()[0]
    
    ## Run ##
    for i in range(300):#steps/6):
        cur_pos = robot_pos[i,1:1+utils.FIELDS].unsqueeze(0)
        mesh_kinematics = utils.concat_mesh_kinematics(mesh, cur_pos)
        correction = net(mesh_kinematics)
        for j in range(ensemble_size):
            update = net(mesh_kinematics)
#        print(update[0,1,:,-1,:])
            correction = correction + update
#    exit()
        network_correction = (correction / (ensemble_size)).detach()
#    print(network_correction[0,0,0])
#    print(mesh[0,2,0,4,4])
        mesh = utils.correct(mesh, network_correction)
#    print(mesh[0,1,:,-1,:])
        write_out = mesh.clone().cpu().numpy().reshape(3,-1).transpose()
        np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)

if __name__ == "__main__":

    data_file = sys.argv[1]

    ## Load kinematics ##
    folder_name = 'data' + str(data_file)
    robot_pos = np.genfromtxt('../../dataset/2019-10-09-GelPhantom1/dvrk/' + folder_name  + '_robot_cartesian_velocity.csv', delimiter=',')
    robot_pos = torch.from_numpy(robot_pos).float().to(device)

    if len(sys.argv) == 3:
        network_path = Path('../deformable/checkpoints/models/model_' + sys.argv[2] + '.pt')
    elif len(sys.argv) < 3:
        network_path = Path('../deformable/augmentation_model.pt')
    else:
        print('Too many arguments')
        exit()
    
    net = SimuNet(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS, dropout=utils.DROPOUT)
    # Load previous model if requested
    if network_path.exists():
        state = torch.load(str(network_path))
        net.load_state_dict(state['model'])
        net = net.to(device)
        print('Restored model')
    else:
        print('Failed to restore model')
        exit()


    ## Load mesh ##
    simulator_file = '../../dataset/2019-10-09-GelPhantom1/simulator/5e3_data/' + folder_name + '/position0001.txt'
    mesh = torch.from_numpy(utils.reshape_volume(np.genfromtxt(simulator_file))).float().unsqueeze(0).to(device)

    play_simulation(net, mesh, robot_pos, folder_name)
