import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


# mesh is from simulator
# pc, or point cloud is from depth camera
def pc_to_mesh_loss(mesh, pc):
    loss = 
    return loss

class MeshLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        pc_to_mesh_loss = pc_to_mesh_loss(input, target)
        # Average the Dice score across all channels/classes
        return torch.mean(loss)
