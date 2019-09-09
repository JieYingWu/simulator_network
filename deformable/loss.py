import torch
import pymesh
import numpy as np

from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, scale, mesh, batch_size, device):
        super(MeshLoss, self).__init__()
        self.normalization = nn.Sigmoid()
        self.mesh = mesh
        self.scale = scale
        self.batch_size = batch_size
        self.device = device
        

    def forward(self, vertices, pc):
        # get probabilities from logits
        loss = torch.zeros(self.batch_size).to(self.device)
        for i in range(self.batch_size):
            cur_v = self.normalization(vertices[i])
            cur_pc = self._scale_points(pc[i])
            loss[i] = self._pc_to_mesh_loss(cur_v, cur_pc)
        # Average the Dice score across all channels/classes
        return torch.mean(loss)

    
    # mesh is from simulator
    # pc, or point cloud is from depth camera
    def _pc_to_mesh_loss(self, vertices, pc):
        pos = vertices.permute(1,2,3,0).view(-1,3)
        pos = pos.detach().cpu().numpy()
        pc = pc.numpy()
        pc = pc[~np.all(pc==0, axis=1)]
        cur_mesh = pymesh.form_mesh(pos, self.mesh.faces, self.mesh.voxels)
        squared_distances, face_indices, closest_points = pymesh.distance_to_mesh(cur_mesh, pc)
        return torch.mean(torch.from_numpy(squared_distances))

    
    def _scale_points(self, pc):
        pc = pc * self.scale
        return pc
