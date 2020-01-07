import os
import torch
import plyfile
import numpy as np
from torch.utils.data import Dataset

def add_gaussian_noise(x, mean=0, stddev=3):
    return x + (torch.randn(x.size()) + mean)* stddev

class SimulatorDataset3D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path, augment=False, pc_length=50000):
        self.pc_length = pc_length
        self.augment= augment
        self.kinematics_array = None
        for path in  kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')))
                
        self.simulator_array = []
        for path in simulator_path:
            files = sorted(os.listdir(path))
            self.simulator_array = self.simulator_array + [path + x for x in files]

        self.label_array = []
        for path in label_path:
            files = sorted(os.listdir(path))
            self.label_array = self.label_array + [path + x for x in files]
        
    def __len__(self):
        return len(self.simulator_array)

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        simulation = torch.from_numpy(self._reshape(np.genfromtxt(self.simulator_array[idx]))).float()
        if self.augment:
            simulation = add_gaussian_noise(simulation)
        pc = plyfile.PlyData.read(self.label_array[idx])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        pc = self._pad(pc)
        pc = np.transpose(pc, (1,0))
        return torch.from_numpy(self.kinematics_array[idx,1:]).float(), simulation, torch.from_numpy(pc).float()

    def _reshape(self, x):
        y = x.reshape(13, 5, 5, 3)
        y = y.transpose((3, 0, 1, 2))        
        return y

    def _pad(self, x):
        padded = np.zeros((self.pc_length, 3))
        padded[0:x.shape[0],:] = x
        return padded

    
class SimulatorDataset2D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path, augment=False, pc_length=30000):
        self.pc_length = pc_length
        self.augment = augment
        self.kinematics_array = None
        for path in  kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')))
                
        self.simulator_array = [] 
        for path in simulator_path:
            files = sorted(os.listdir(path))
            self.simulator_array = self.simulator_array + [path + x for x in files]
        self.label_array = []
        
        for path in label_path:
            files = sorted(os.listdir(path))
            self.label_array = self.label_array + [path + x for x in files]
        
    def __len__(self):
        return len(self.simulator_array)

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
#        simulation_time = time()
        simulation = np.genfromtxt(self.simulator_array[idx])
        simulation = torch.from_numpy(self._reshape(simulation)).float()
        if self.augment:
            simulation  = add_gaussian_noise(simulation)
#        print('Loading simulation took ' + str(time()-simulations_time) + ' s')

#        label_time = time()
        pc = plyfile.PlyData.read(self.label_array[idx])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        pc = self._truncate(pc)
        pc = np.transpose(pc, (1,0))
#        print('Loading label took ' + str(time()-label_time) + ' s')        
         
        return torch.from_numpy(self.kinematics_array[idx,1:]).float(), simulation, torch.from_numpy(pc).float()

    def _reshape(self, x):
        y = x.reshape(13, 5, 5, 3)
        y = y.transpose((3, 0, 1, 2))
        y = y[:,:,-1,:]
        return y

    def _pad(self, x):
        padded = np.zeros((self.pc_length, 3))
        padded[0:x.shape[0],:] = x
        return padded

    def _truncate(self, x):
        indices = torch.randperm(30000)
        truncated = x[indices, :]
        return truncated
