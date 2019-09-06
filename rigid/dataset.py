import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class SimulatorDataset(Dataset):
    def __init__(self, path):
        self.kinematics_array = None
        self.simulator_array = None
        self.label_array = None
        for p in  path:
            if self.kinematics_array is None:
                kinematics_array = np.genfromtxt(p+'_robot_cartesian_processed.csv', delimiter=',')
                simulator_array = np.genfromtxt(p+'_cartesian_simulation.csv', delimiter=',')
                label_array = np.genfromtxt(p+'_polaris_processed.csv', delimiter=',')
                self.kinematics_array = kinematics_array
                self.simulator_array = simulator_array
                self.label_array = self._interpolate(kinematics_array, label_array)
            else:
                kinematics_array = np.genfromtxt(p+'_robot_cartesian_processed.csv', delimiter=',')
                simulator_array = np.genfromtxt(p+'_cartesian_simulation.csv', delimiter=',')
                label_array = np.genfromtxt(+'_polaris_processed.csv', delimiter=',')
                self.kinematics_array = np.concatenate((self.kinematics_array, kinematics_array))
                self.simulator_array = np.concatenate((self.simulator_array, simulator_array))
                self.label_array = np.concatenate((self.label_array, self._interpolate(kinematics_array, label_array)))
        
    def __len__(self):
        return self.kinematics_array.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.append(self.kinematics_array[idx,1:], self.simulator_array[idx,1:])).float(), torch.from_numpy(self.interpolated_labels[idx,1:]).float()

    def _interpolate(self, kinematics_array, label_array):
        label_time = label_array[:,0]
        kinematics_time = kinematics_array[:,0]
        new_label_array = np.zeros(kinematics_array.shape)
        new_label_array[:,0] = kinematics_time
        for j in range(1,self.kinematics_array.shape[1]):
            new_label_array[:,j] = np.interp(kinematics_time, label_time, label_array[:,j])
        return new_label_array
