import torch
import numpy as np
from torch import nn as nn
from chamferdist.chamferdist import ChamferDistance

import sys
np.set_printoptions(threshold=sys.maxsize)

def refine_mesh(mesh, factor, device):
    batch,z,x,y =  mesh.size()
    new_x = (x-1)*factor+1
    new_y = (y-1)*factor+1
    fine_mesh = torch.zeros([batch, z, new_x, new_y], device=device)

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


class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, batch_size, weight, device):
        super(MeshLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()
        self.fem_loss_fn = nn.MSELoss()
        self.weight = weight

    def forward(self, network_mesh, pc, fem_mesh):
        # get probabilities from logits
        top = network_mesh[:,:,:,-1,:]
        bottom = network_mesh[:,:,:,0:-1,:]
        fem = fem_mesh[:,:,:,0:-1,:]
        
        # Match the top, camera observed layer
        top = refine_mesh(top, 3, self.device)
        top = top.reshape(top.size()[0],top.size()[1],-1)
        
        pc = pc.permute(0,2,1).contiguous()
        top = top.permute(0,2,1).contiguous()
        dist1, dist2, idx1, idx2 = self.chamfer(top, pc)

        if False:
            num = 10
            print('--------------------------')
            idx = idx1[0,0:num].detach().cpu().numpy()
            dist = dist1[0,0:num].detach().cpu().numpy()
            print(idx)
            print(dist)
            print(pc[0].detach().cpu().numpy()[idx])
            print(top[0,0:num])
            print(fem[0,0,0,:,:])
            exit()

        # Match the bottom, FEM layers
        fem_loss = self.fem_loss_fn(network_mesh, fem_mesh)
        # Only want pc -> mesh loss to ignore occluded regions
#        print(torch.mean(dist1), fem_loss*self.weight)
        loss = torch.mean(dist2) + fem_loss * self.weight + torch.mean(dist1)
        # Average the Dice score across all channels/classes
        return loss


class MeshLoss2D(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self,  batch_size, device):
        super(MeshLoss2D, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()

    def forward(self, network_mesh, pc):
        # get probabilities from logits
        loss = torch.zeros(network_mesh.size()[0]).to(self.device)

        mesh = refine_mesh(network_mesh, 3, self.devic)
        mesh = mesh.reshape(mesh.size()[0],mesh.size()[1],-1)
        mesh = mesh.permute(0,2,1).contiguous()
        pc = pc.permute(0,2,1).contiguous()
        
        dist1, dist2, idx1, idx2 = self.chamfer(mesh, pc)

        # Average the Dice score across all channels/classes
        return torch.mean(dist2)
 
