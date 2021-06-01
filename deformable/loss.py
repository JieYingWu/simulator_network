import torch
import numpy as np
from torch import nn as nn
from utils import refine_mesh
#from chamfer3D.dist_chamfer_3D import chamfer_3DDist
from chamferdist import ChamferDistance
# chamfer dist from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch

import sys
np.set_printoptions(threshold=sys.maxsize)

class RegularizationLoss(nn.Module):

    def __init__(self):
        super(RegularizationLoss, self).__init__()
        self.distance_fn = nn.MSELoss()
        
    def forward(self, mesh):
        size = mesh.size()
        loss = 0
        for x in range(mesh.size()[2]-1):
            for y in range(mesh.size()[3]-1):
                for z in range(mesh.size()[4]-1):
                    loss += self.distance_fn(mesh[:,:,x,y,z], mesh[:,:,x+1,y,z])
                    loss += self.distance_fn(mesh[:,:,x,y,z], mesh[:,:,x,y+1,z])
                    loss += self.distance_fn(mesh[:,:,x,y,z], mesh[:,:,x,y,z+1])
        return loss
    
class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, fem_weight, reg_weight, device):
        super(MeshLoss, self).__init__()
        self.device = device
        self.chamfer = ChamferDistance()
        self.fem_loss_fn = nn.MSELoss()
        self.fem_weight = fem_weight
        self.reg_weight = reg_weight
        self.reg_fn = RegularizationLoss()

    def forward(self, pred, pc, fem_mesh):

        # if pc is not None:
        #     # Match the top, camera observed layer
        #     top = network_mesh[:,:,:,-1,:]
        #     #        top = refine_mesh(top, 3, self.device)
        #     top = top.reshape(top.size()[0],top.size()[1],-1)
        
        #     pc = pc.permute(0,2,1).contiguous()
        #     top = top.permute(0,2,1).contiguous()
        #     dist1, dist2, idx1, idx2 = self.chamfer(top, pc)
        # else:
        #     dist2 = torch.zeros(1)

        fem_loss = self.fem_loss_fn(pred, fem_mesh)
        regularization = self.reg_fn(pred)
        # Only want pc -> mesh loss to ignore occluded regions
        loss = fem_loss*self.fem_weight + regularization*self.reg_weight# + torch.mean(dist2)# + torch.mean(dist1)
        #print(fem_loss, regularization, torch.mean(dist2))
#        print(torch.mean(dist2), fem_loss)
        return loss


class MeshLoss2D(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self,  batch_size, device):
        super(MeshLoss2D, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance() #chamfer_3DDist()

    def forward(self, vertices, pc):
        # get probabilities from logits
        #fine_mesh = vertices
        vertices = refine_mesh(vertices, 2, self.device)
        n = vertices.size()[0]
        top = vertices.reshape(n,3,-1)
        top = top.permute(0, 2, 1)
#        top = top[:,0,:].unsqueeze(1)
#        pc = pc[:,0,:].unsqueeze(1)
        
        dist= self.chamfer(pc, top)
 #       print(dist1)
 #       print(idx1)
            # Only want pc -> mesh loss to ignore occluded regions
 
        # Average the Dice score across all channels/classes
#        return torch.mean(loss)
        return dist
