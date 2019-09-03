import torch
from torch import nn 
from numpy import dot


class RigidLoss(nn.Module):
    """Computes loss from a point cloud to a mesh
    """

    def __init__(self, weight):
        super(RigidLoss, self).__init__()
        self.weight = weight
        self.translation_loss = nn.MSELoss()


    def forward(self, input, target):
        t = self.translation_loss(input[:,0:3], target[:,0:3])
        q = self._quaternion_loss(input[:,3:7], target[:,3:7])
        return torch.mean(t + self.weight * q)


    def _quaternion_loss(self, input, target):
        length = input.size()[0]
        input_norm = torch.sqrt(torch.sum(torch.mul(input,input), dim=1))
        target_norm = torch.sqrt(torch.sum(torch.mul(target, target), dim=1))
        input = input/input_norm.view(length,1)
        target = target/target_norm.view(length,1)
        dot = torch.abs(torch.bmm(input.reshape(length,1,4), target.reshape(length,4,1)))
        loss = 2*torch.acos(dot)
        return loss
