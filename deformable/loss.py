import torch
import numpy as np
from torch import nn as nn
from chamferdist.chamferdist import ChamferDistance

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

        # Match the bottom, FEM layers
        fem_loss = self.fem_loss_fn(bottom, fem)
        # Only want pc -> mesh loss to ignore occluded regions
#        print(torch.mean(dist1), fem_loss*self.weight)
        loss = torch.mean(dist2) + fem_loss * self.weight# + torch.mean(dist1)
        
        # Average the Dice score across all channels/classes
        return torch.mean(loss).to(self.device)


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
        for i in range(network_mesh.size()[0]):            
            cur_v = network_mesh[i]
            cur_v = refine_mesh(cur_v, 3, self.device)
            cur_v = cur_v.reshape(3,-1)
            cur_v = torch.transpose(cur_v, 0,1).unsqueeze(0)

            cur_pc = pc[i]
            cur_pc = cur_pc[:,~(cur_pc==0).all(axis=0)]
            cur_pc = torch.transpose(cur_pc, 0,1).unsqueeze(0)

            dist1, dist2, idx1, idx2 = self.chamfer(cur_v.contiguous(), cur_pc.continguous())

            # Only want pc -> mesh loss to ignore occluded regions
            loss[i] = torch.mean(dist2)

        # Average the Dice score across all channels/classes
        return torch.mean(loss)
 
