import torch
import numpy as np
from torch import nn as nn
from utils import refine_mesh
from chamferdist.chamferdist import ChamferDistance

import sys
np.set_printoptions(threshold=sys.maxsize)


class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, batch_size, weight, base_mesh, device):
        super(MeshLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()
        self.fem_loss_fn = nn.MSELoss()
        self.weight = weight
        self.base_mesh = base_mesh

    def forward(self, network_mesh, pc, fem_mesh):
        # get probabilities from logits
        top = network_mesh[:,:,:,-1,:]
        bottom = network_mesh[:,:,:,0:-1,:]
        
        # Match the top, camera observed layer
#        top = refine_mesh(top, 3, self.device)
        top = top.reshape(top.size()[0],top.size()[1],-1)
        
        pc = pc.permute(0,2,1).contiguous()
        top = top.permute(0,2,1).contiguous()
#        print('-------')
#        print(top)
#        print(pc)
        dist1, dist2, idx1, idx2 = self.chamfer(top, pc)

        if False:
            num = 10
            print('--------------------------')
#            idx = idx2[0,0:num].detach().cpu().numpy()
#            dist = dist2[0,0:num].detach().cpu().numpy()
            dist1 = dist1[0].detach().cpu().numpy()
            idx1 = idx1[0].detach().cpu().numpy()
            dist2 = dist2[0].detach().cpu().numpy()
            idx2 = idx2[0].detach().cpu().numpy()
            print(torch.max(pc[0,:,1]))
            print(torch.min(pc[0,:,1]))
            print(dist1.shape, idx1.shape)
            pt = idx1[np.argmax(dist1)]
            print(np.max(dist1))
            print(pc[0,pt])
            print(top[0,np.argmax(dist1)])
            pt = idx2[np.argmax(dist2)]
            print(np.max(dist2))
            print(top[0,pt])
            print(pc[0,np.argmax(dist2)])
#            print(top[0,0:num])
#            print(fem[0,0,0,:,:])
            exit()

        # Match the FEM layers
#        print(base_mesh[0,2,:,1,:])
#        print(network_mesh[0,2,:,1,:])
        fem_loss = self.fem_loss_fn(bottom, fem_mesh[:,:,:,0:-1,:])
        # Only want pc -> mesh loss to ignore occluded regions
        loss = torch.mean(dist2) + fem_loss * self.weight# + torch.mean(dist1)
        #print(torch.mean(dist2), fem_loss)
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
 
