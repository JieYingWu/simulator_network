import os
import torch
import utils
import random
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import torch_geometric.transforms as T
from model import GraphUNet, BSplineNet
from torch_geometric.data import Dataset

augment_steps = 1
scale = utils.scale.numpy()/10

class MeshGraphDataset(Dataset):
    def __init__(self, kinematics_path, fem_path, augment=False):
        super(MeshGraphDataset, self).__init__(None, None, None)
        self.kinematics_array = None
        for path in kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')[:,1:utils.FIELDS+1]
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')[:,1:utils.FIELDS+1]))
        self.kinematics_array = np.concatenate((self.kinematics_array[:,0:3], self.kinematics_array[:,7:10]), axis=1)
        self.kinematics_array = torch.from_numpy(self.kinematics_array).float()

#        self.fem_array = []
        self.all_fem = None
        for path in fem_path:
            print(path)
            files = sorted(os.listdir(path))            
#            temp = [path + x for x in files]
#            self.fem_array = self.fem_array + temp
            for x in files:
                fem = np.genfromtxt(path+x)
                fem = torch.from_numpy(fem).float()
                fem = fem.unsqueeze(0)
                if self.all_fem is None:
                    self.all_fem = fem
                else:
                    self.all_fem = torch.cat((self.all_fem, fem), axis=0)
            
        base_mesh = 'mesh_fine.txt'
        self.base_mesh = torch.from_numpy(np.genfromtxt(base_mesh)).float()

        self.edge_index = utils.make_edges()
        self.augment = augment
        
    def __set_network__(self):
        network_path = Path('augmentation_model.pt')
        self.net = GraphUNet(utils.IN_CHANNELS, 256, 3, utils.NET_DEPTH, sum_res=False)
        # Load previous model if requested
        if network_path.exists():
            state = torch.load(str(network_path), map_location='cpu')
            self.net.load_state_dict(state['model'])
            del state
#            print('Restored model')
        else:
            print('Failed to restore model for augmentation')
            self.net = False

    def add_gaussian_noise(self, mesh):
        x = torch.empty(mesh.size()[0], device='cpu').normal_(mean=0,std=scale[0]).unsqueeze(1)
        y = torch.empty(mesh.size()[0], device='cpu').normal_(mean=0,std=scale[1]).unsqueeze(1)
        z = torch.empty(mesh.size()[0], device='cpu').normal_(mean=0,std=scale[2]).unsqueeze(1)
        modifier = torch.cat((x,y,z), axis=1)
        return mesh + modifier

    def add_model_noise(self, idx, steps):
        if idx < steps:
            idx = steps
            
#        simulation_prev = np.genfromtxt(self.fem_array[idx - steps])
#        simulation_prev = torch.from_numpy(simulation_prev).float()
        simulation_prev = self.all_fem[idx-steps]
            
        for i in range(steps,0,-1):
            kinematics_prev = self.kinematics_array[idx-i].unsqueeze(0)
            kinematics_arr = torch.zeros(simulation_prev.size()[0], 6).float()
            closest_point = self.closest_node(kinematics_prev[:,0:3], simulation_prev)
            kinematics_arr[closest_point, :] = kinematics_prev
            correction = self.net(torch.cat((simulation_prev, kinematics_arr), axis=1), self.edge_index)
            simulation_prev = simulation_prev + utils.correct(simulation_prev, correction.detach(), 'cpu')

        return simulation_prev.squeeze(0)
            
    def len(self):
#        return len(self.fem_array)-1
        return self.all_fem.size()[0]-1
    
    # return robot kinematics, mesh, and point cloud
    def get(self, idx):
#        if self.augment and random.random() < 0.5:
#            fem = np.genfromtxt(self.fem_array[idx])
#            fem = torch.from_numpy(fem).float()
#            fem = self.add_gaussian_noise(fem)
        if self.augment and self.net and random.random() < 0.5:
            value = random.randint(0, augment_steps)
            fem = self.add_model_noise(idx, value)
        else:
            fem = self.all_fem[idx]
#            fem = torch.from_numpy(fem).float()

        kinematics = self.kinematics_array[idx].unsqueeze(0)
        kinematics_arr = torch.zeros(fem.size()[0], 6).float()
        closest_point = self.closest_node(kinematics[:,0:3], fem)
        if closest_point is not None:
            kinematics[:,0:3] = kinematics[:,0:3] - fem[closest_point,:]
            kinematics_arr[closest_point, :] = kinematics
        
        fem_next = self.all_fem[idx+1]
        corr = fem_next - fem
        fem = torch.cat((fem-self.base_mesh, kinematics_arr), axis=1)
        edge_attr = fem[self.edge_index[1]] - fem[self.edge_index[0]]
#        edge_attr = fem[self.edge_index[1]][0:3] - fem[self.edge_index[0]][0:3]
        fem = Data(x=fem, edge_index=self.edge_index)# ,edge_attr=edge_attr.type(torch.float32))
        corr = Data(x=corr, edge_index=self.edge_index)


        return fem, corr#, pc_last

    def closest_node(self, node, nodes):
        dist2 = torch.sum((nodes - node)**2, axis=1)
        min_value = torch.min(dist2)
        if min_value > utils.MAX_DIST:
            min_index = None
        else:
            min_index = torch.argmin(dist2)
        return min_index
