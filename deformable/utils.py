import json
import torch
from torch import nn
from datetime import datetime


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal"):
    assert (torch.cuda.is_available())
    net = net.cuda()
    kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net

def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    for module in model.modules():
        if hasattr(module, 'weight'):
            print(module.weight)
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
scale[0,:,0,0,0] = torch.tensor([5.28, 7.16, 7.86])/2
def correct(mesh,x):
    x = (x-0.5)*scale
    corrected = mesh + x
    return corrected

scale_cpu = torch.zeros((1,3,1,1,1))
scale_cpu[0,:,0,0,0] = torch.tensor([5.28, 7.16, 7.86])/2
def correct_cpu(mesh,x):
    x = (x-0.5)*scale_cpu
    corrected = mesh + x
    return corrected


def concat_mesh_kinematics(mesh, kinematics):
    kinematics = kinematics.view(kinematics.size()[0], kinematics.size()[1],1,1,1)
    kinematics = kinematics.repeat(1,1,mesh.size()[2],mesh.size()[3],mesh.size()[4])
#    print(mesh.size())
#    print(kinematics.size())
    return torch.cat((mesh, kinematics), axis=1)

def reshape_volume(x):
    y = x.reshape(13, 5, 5, 3)
    y = y.transpose((3, 0, 1, 2))        
    return y



#def write_event(log, step: int, **data):
#    data['step'] = step
#    data['dt'] = datetime.now().isoformat()
#    log.write(json.dumps(data, sort_keys=True))
#    log.write('\n')
#    log.flush()
