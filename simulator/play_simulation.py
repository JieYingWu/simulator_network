import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0,'../processing/')
sys.path.insert(0,'../deformable/')
import geometry_util as geo
from model import UNet3D

import gc

all_time_steps = [0.0332, 0.0332, 0.0329, 0.0332, 0.0332, 0.0333, 0.0331, 0.0332, 0.0332, 0.0328, 0.0455, 0.0473] 
data_file = sys.argv[1]
folder_name = 'data' + str(data_file)

device = torch.device('cuda') 
scale = torch.zeros((1,3,1,1,1))
scale[0,:,0,0,0] = torch.tensor([5.28, 7.16, 7.86])/2
scale = scale.to(device)

def correct(mesh,x):
    x = (x-0.5)*scale
    corrected = mesh + x
    return corrected

def reshape(x):
    y = x.reshape(13, 5, 5, 3)
    y = y.transpose((3, 0, 1, 2))        
    return y

## Load network ##
in_channels = 3
out_channels = 3
network_path = Path('../deformable/checkpoints/models/model_' + sys.argv[2] + '.pt')

net = UNet3D(in_channels=in_channels, out_channels=out_channels)
# Load previous model if requested
if network_path.exists():
    state = torch.load(str(network_path))
    net.load_state_dict(state['model'])
    net = net.to(device)
    print('Restored model')
else:
    print('Failed to restore model')
    exit()
net.eval()

## Load kinematics ##
robot_pos = np.genfromtxt('../../dataset/2019-10-09-GelPhantom1/dvrk/' + folder_name  + '_robot_cartesian_processed_interpolated.csv', delimiter=',')
robot_pos = torch.from_numpy(robot_pos).float().to(device)
steps = robot_pos.size()[0]

## Load mesh ##
simulator_file = '../../dataset/2019-10-09-GelPhantom1/simulator/5e3_data/' + folder_name + '/position0001.txt'
mesh = torch.from_numpy(reshape(np.genfromtxt(simulator_file))).float().unsqueeze(0).to(device)
        
## Run ##
for i in range(steps):
    cur_pos = robot_pos[i,1:8].unsqueeze(0)
    network_correction = net(cur_pos, mesh)
    mesh = correct(mesh, network_correction)
    write_out = mesh.clone().detach().cpu().numpy().reshape(3,-1).transpose()
    np.savetxt(folder_name + "/position" + '%04d' % (i) + ".txt", write_out)
    del cur_pos, network_correction, write_out
    # print('----------------------------------')
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except:
    #         pass
