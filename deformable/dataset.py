import os
import torch
import plyfile
import numpy as np
from torch.utils.data import Dataset

shape = (3, 13, 5, 5)
pc_length = 16000

class SimulatorDataset3D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path):
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
#        print(self.kinematics_array.shape[0], len(self.simulator_array), len(self.label_array))
        return len(self.simulator_array)

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        simulation = np.genfromtxt(self.simulator_array[idx])
        pc = plyfile.PlyData.read(self.label_array[idx])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        pc = self._pad(pc)
        pc = np.transpose(pc, (1,0))
        return torch.from_numpy(self.kinematics_array[idx,1:]).float(), torch.from_numpy(self._reshape(simulation)).float(), torch.from_numpy(pc).float()

    def _reshape(self, x):
        idx = 0
        y = np.zeros(shape)
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in range(shape[3]):
                    y[:,i,j,k] = x[idx]
        return y


    def _pad(self, x):
        padded = np.zeros((pc_length, 3))
        padded[0:x.shape[0],:] = x
        return padded
