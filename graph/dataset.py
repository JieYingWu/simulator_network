import os
import torch
import utils
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import Dataset

class MeshDataset(torch.utils.data.Dataset):
    def __init__(self, kinematics_path, fem_path):
        self.kinematics_array = None
        for path in kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')[0:-1,1:utils.FIELDS+1]
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')[0:-1,1:utils.FIELDS+1]))
        self.kinematics_array = np.concatenate((self.kinematics_array[:,0:3], self.kinematics_array[:,7:10]), axis=1)
        self.kinematics_array = torch.from_numpy(self.kinematics_array).float()
                
        self.fem_array = []
        for path in fem_path:
            files = sorted(os.listdir(path))
            self.fem_array = self.fem_array + [path + x for x in files]

        base_mesh = 'mesh_fine.txt'
        self.base_mesh = torch.from_numpy(np.genfromtxt(base_mesh)).float()

            
    def __len__(self):
        return len(self.fem_array)-1

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        fem = np.genfromtxt(self.fem_array[idx])
        fem = torch.from_numpy(fem).float()

        kinematics = self.kinematics_array[idx].unsqueeze(0)
        kinematics_arr = torch.zeros(fem.size()[0], 6).float()
        closest_point = self.closest_node(kinematics[:,0:3], fem)
        kinematics_arr[closest_point, :] = kinematics
        
        fem_next = torch.from_numpy(np.genfromtxt(self.fem_array[idx+1])).float()
        fem_next = fem_next- fem
        fem = torch.cat((fem, kinematics_arr), axis=1)
        return kinematics, fem, fem_next#, pc_last

    def closest_node(self, node, nodes):
        dist_2 = torch.sum((nodes - node)**2, axis=1)
        return torch.argmin(dist_2)


class MeshGraphDataset(Dataset):
    def __init__(self, kinematics_path, fem_path):
        super(MeshGraphDataset, self).__init__(None, None, None)
        self.kinematics_array = None
        for path in kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')[0:-1,1:utils.FIELDS+1]
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')[0:-1,1:utils.FIELDS+1]))
        self.kinematics_array = np.concatenate((self.kinematics_array[:,0:3], self.kinematics_array[:,7:10]), axis=1)
        self.kinematics_array = torch.from_numpy(self.kinematics_array).float()
                
        self.fem_array = []
        for path in fem_path:
            files = sorted(os.listdir(path))
            self.fem_array = self.fem_array + [path + x for x in files]

        base_mesh = 'mesh_fine.txt'
        self.base_mesh = torch.from_numpy(np.genfromtxt(base_mesh)).float()

        self.edge_index = utils.make_edges()

            
    def len(self):
        return len(self.fem_array)-1

    # return robot kinematics, mesh, and point cloud
    def get(self, idx):
        fem = np.genfromtxt(self.fem_array[idx])
        fem = torch.from_numpy(fem).float()

        kinematics = self.kinematics_array[idx].unsqueeze(0)
        kinematics_arr = torch.zeros(fem.size()[0], 6).float()
        closest_point = self.closest_node(kinematics[:,0:3], fem)
        kinematics_arr[closest_point, :] = kinematics
        
        fem_next = torch.from_numpy(np.genfromtxt(self.fem_array[idx+1])).float()
        fem_next = fem_next- fem
        fem = torch.cat((fem, kinematics_arr), axis=1)
        fem = Data(x=fem, edge_index=self.edge_index)
        fem_next = Data(x=fem_next, edge_index=self.edge_index)        
        return fem, fem_next#, pc_last

    def closest_node(self, node, nodes):
        dist_2 = torch.sum((nodes - node)**2, axis=1)
        return torch.argmin(dist_2)
