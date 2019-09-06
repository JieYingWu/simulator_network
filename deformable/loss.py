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


    
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
 
    def forward(self,preds,gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)
		#brk()
		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P
