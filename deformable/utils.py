import json
import torch
import numpy as np
from torch import nn
from datetime import datetime


FIELDS = 10
IN_CHANNELS = 3
OUT_CHANNELS = 3
DROPOUT = 0#.01
VOL_SIZE = np.array([25,9,9,3])
#VOL_SIZE = np.array([13,5,5,3])
VOL_DIM = np.array([68.7, 35.8, 39.3])

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scale = torch.zeros((1,3,1,1,1), device=device)
scale[0,:,0,0,0] = torch.tensor(VOL_DIM/VOL_SIZE[0:3])
def correct(mesh,x):
    x = x - 0.5
    x = x*scale
    corrected = mesh + x
    return corrected

scale_cpu = scale.cpu()
def correct_cpu(mesh,x):
    x = x - 0.5
    x = x*scale_cpu
    corrected = mesh + x
    return corrected

def concat_mesh_kinematics(mesh, kinematics):
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

def refine_mesh(mesh, factor, device):
    batch,n,x,y =  mesh.size()
    new_x = (x-1)*factor+1
    new_y = (y-1)*factor+1
    fine_mesh = torch.zeros([batch, n, new_x, new_y], device=device)

    # Fill out the rows 
    for b in range(batch):
        for i in range(x):
            cur_row = mesh[b,:,i,:]
            for j in range(y-1):
                cur_elem = cur_row[:,j]
                next_elem = cur_row[:,j+1]
                for k in range(factor):
                    weight = float(k)
                    fine_mesh[b,:,i*factor,j*factor+k] = (factor-weight)/factor*cur_elem + weight/factor*next_elem
                    fine_mesh[b,:,i*factor,-1] = mesh[b,:,i,-1]
                
        # Fill out the columns
        for j in range(new_y):
            cur_col = fine_mesh[b,:,:,j]
            for i in range(x-1):
                cur_elem = cur_col[:,i*factor]
                next_elem = cur_col[:,(i+1)*factor]
                for k in range(factor):
                    weight = float(k)
                    fine_mesh[b,:,i*factor+k,j] = (factor-weight)/factor*cur_elem + weight/factor*next_elem

    return fine_mesh

#def write_event(log, step: int, **data):
#    data['step'] = step
#    data['dt'] = datetime.now().isoformat()
#    log.write(json.dumps(data, sort_keys=True))
#    log.write('\n')
#    log.flush()


