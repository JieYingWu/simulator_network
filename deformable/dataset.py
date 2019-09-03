import os
import torch
import pymesh
import numpy as np
from torch.utils.data import Dataset

shape = (3, 13, 5, 5)
pc_length = 270000

class SimulatorDataset3D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path):
        self.kinematics_path = kinematics_path
        self.kinematics_array = np.genfromtxt(kinematics_path, delimiter=',')
        self.simulator_path = simulator_path
        self.simulator_array = os.listdir(simulator_path)
        self.label_path = label_path
        self.label_array = os.listdir(label_path) # Only has the file name, not path
        #self.interpolate()
        
    def __len__(self):
#        print(self.kinematics_array.shape[0], len(self.simulator_array), len(self.label_array))
        return 364#self.kinematics_array.shape[0]

    # return robot kinematics, pymesh mesh, and point cloud
    def __getitem__(self, idx):
        try:
            simulation = np.genfromtxt(self.simulator_path + self.simulator_array[idx])
        except:
            print('error file is ' + self.simulator_array[idx])
            exit()
        pc = pymesh.load_mesh(self.label_path + self.label_array[idx]).vertices
        pc = self._pad(pc)
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

        
#    def interpolate(self):
#        label_time = [int(os.path.splitext(x)[0]) for x in self.label_array]
#        kinematics_time = self.kinematics_array[:,0]
#        new_kinematics_array = np.zeros(self.kinematics_array.shape)
#        new_kinematics_array[:,0] = label_time
#        for j in range(1,self.kinematics_array.shape[1]):
#            new_kinematics_array[:,j] = np.interp(label_time, kinematics_time, self.kinematics_array[:,j])
#        self.interpolated_kinematics = new_kinematics_array
#        return 0
