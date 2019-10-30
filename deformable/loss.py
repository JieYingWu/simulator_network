import torch
import numpy as np
from torch import nn as nn
from chamfer_distance.chamfer_distance import ChamferDistance

def refine_mesh(mesh, factor):
    batch,z,x,y =  mesh.size()
    new_x = (x-1)*factor+1
    new_y = (y-1)*factor+1
    fine_mesh = torch.zeros([batch, z, new_x, new_y])

    # Fill out each mesh
    for n in range(batch):
        # Fill out the rows 
        for i in range(x):
            cur_row = mesh[n,:,i,:]
            for j in range(y-1):
                cur_elem = cur_row[:,j]
                next_elem = cur_row[:,j+1]
                for k in range(factor):
                    weight = float(k)
                    fine_mesh[n,:,i*factor,j*factor+k] = (factor-weight)/factor*cur_elem + weight/factor*next_elem
        fine_mesh[n,:,i*factor,-1] = mesh[n,:,i,-1]
                
        # Fill out the columns
        for j in range(new_y):
            cur_col = fine_mesh[n,:,:,j]
            for i in range(x-1):
                cur_elem = cur_col[:,i*factor]
                next_elem = cur_col[:,(i+1)*factor]
                for k in range(factor):
                    weight = float(k)
                    fine_mesh[n,:,i*factor+k,j] = (factor-weight)/factor*cur_elem + weight/factor*next_elem

    return fine_mesh


class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, batch_size, device):
        super(MeshLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()

    def forward(self, vertices, pc):
        # get probabilities from logits
        loss = torch.zeros(vertices.size()[0])
        top = vertices[:,:,:,-1,:]
        top = refine_mesh(top, 3)
        top = top.reshape(vertices.size()[0],3,-1)
        for i in range(vertices.size()[0]):
            cur_v = top[i]
            cur_v = torch.transpose(cur_v, 0,1).unsqueeze(0)
            cur_pc = pc[i]
            cur_pc = cur_pc[:,~(cur_pc==0).all(axis=0)]
            cur_pc = torch.transpose(cur_pc, 0,1).unsqueeze(0)

            dist1, dist2 = self.chamfer(cur_v.cpu(), cur_pc.cpu())

            # Only want pc -> mesh loss to ignore occluded regions
            loss[i] = torch.mean(dist2) # + torch.mean(dist1)

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

    def forward(self, vertices, pc):
        # get probabilities from logits
        loss = torch.zeros(vertices.size()[0]).to(self.device)
        top = vertices.reshape(vertices.size()[0],3,-1)
        for i in range(vertices.size()[0]):
            cur_v = top[i]
            cur_v = torch.transpose(cur_v, 0,1).unsqueeze(0)
            cur_pc = pc[i]
            cur_pc = cur_pc[:,~(cur_pc==0).all(axis=0)]
            cur_pc = torch.transpose(cur_pc, 0,1).unsqueeze(0)

            dist1, dist2 = self.chamfer(cur_v.cpu(), cur_pc.cpu())

            # Only want pc -> mesh loss to ignore occluded regions
            loss[i] = torch.mean(dist2)

        # Average the Dice score across all channels/classes
        return torch.mean(loss)
