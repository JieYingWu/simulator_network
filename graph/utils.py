import json
import torch
import numpy as np
from torch import nn
from datetime import datetime


FIELDS = 10
IN_CHANNELS = 9
OUT_CHANNELS = 3
NET_DEPTH = 2
DROPOUT = 0#.01
VOL_SIZE = np.array([25,9,9,3])
MAX_DIST = 100
DIM = np.array([68.7, 35.8, 39.3], dtype=np.float32)
scale = torch.tensor(DIM/VOL_SIZE[0:3]).type(torch.float32)#*2

def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net

def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                if distribution == "uniform":
                    nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

def correct(mesh,x,device):
    x = x - 0.5
    x = x*scale.to(device)
#    corrected = mesh + x
    return x#corrected

def concat_mesh_kinematics(mesh, kinematics, device):
    kinematics = kinematics.view(kinematics.size()[0], kinematics.size()[1],1,1,1)
    # Just putting it on the top of the mesh
    kinematics = kinematics.repeat(1,1,mesh.size()[2],1,mesh.size()[4])
    zeros = torch.zeros((kinematics.size()[0], kinematics.size()[1], mesh.size()[2], mesh.size()[3]-1, mesh.size()[4]), device=mesh.device)
    kinematics = torch.cat((zeros, kinematics), axis=3)
    return torch.cat((mesh, kinematics), axis=1)

def reshape_volume(x):
    y = x.reshape(VOL_SIZE)
    y = y.transpose((3, 0, 1, 2))        
    return y


def make_edges():
    start = []
    end = []
    nodes = torch.arange(2025).reshape(25,9,9)
        
    def add_undirected_edge(i, j, k, i_n, j_n, k_n):
        start.append(nodes[i,j,k])
        end.append(nodes[i_n,j_n,k_n])
        start.append(nodes[i_n,j_n,k_n])
        end.append(nodes[i,j,k]) 
        
    for i in range(VOL_SIZE[0]):
        for j in range(VOL_SIZE[1]):
            for k in range(VOL_SIZE[2]):
                if (i < VOL_SIZE[0]-1):
                    add_undirected_edge(i,j,k,i+1,j,k)
                if (j < VOL_SIZE[1]-1):
                    add_undirected_edge(i,j,k,i,j+1,k)
                if (k < VOL_SIZE[2]-1):
                    add_undirected_edge(i,j,k,i,j,k+1)                
             
    edge_index = torch.tensor([start, end], dtype=torch.long)
    return edge_index
