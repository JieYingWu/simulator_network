import torch
import numpy as np
from torch import nn as nn
from chamfer_distance.chamfer_distance import ChamferDistance


class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, scale, offset, batch_size, device):
        super(MeshLoss, self).__init__()
        self.scale = scale
        self.offset = offset
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()

    def forward(self, vertices, pc):
        # get probabilities from logits
        loss = torch.zeros(vertices.size()[0])
        top = vertices[:,:,:,-1,:].reshape(vertices.size()[0],3,-1)
        top = (top-self.offset)*self.scale
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

    def __init__(self, scale, offset, batch_size, device):
        super(MeshLoss2D, self).__init__()
        self.scale = scale
        self.offset = offset
        self.batch_size = batch_size
        self.device = device
        self.chamfer = ChamferDistance()

    def forward(self, vertices, pc):
        # get probabilities from logits
        loss = torch.zeros(vertices.size()[0]).to(self.device)
        top = vertices.reshape(vertices.size()[0],3,-1)
        top = (top-self.offset)*self.scale
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
