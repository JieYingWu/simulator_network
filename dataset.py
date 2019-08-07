import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class SimulatorDataset(Dataset):
    def __init__(self, kinematics_file, simulator_file, label_file):
        self.kinematics_file = kinematics_file
        self.kinematics_array = np.genfromtxt(kinematics_file, delimiter=',')
        self.simulator_file = simulator_file
        self.simulator_array = np.genfromtxt(simulator_file, delimiter=',')
        self.label_file = label_file
        self.label_array = np.genfromtxt(label_file, delimiter=',')
        self.interpolate()
        
    def __len__(self):
        return self.kinematics_array.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.append(self.kinematics_array[idx,1:], self.simulator_array[idx,1:])).float(), torch.from_numpy(self.interpolated_labels[idx,1:]).float()

    def interpolate(self):
        label_time = self.label_array[:,0]
        kinematics_time = self.kinematics_array[:,0]
        new_label_array = np.zeros(self.kinematics_array.shape)
        new_label_array[:,0] = kinematics_time
        for j in range(1,self.kinematics_array.shape[1]):
            new_label_array[:,j] = np.interp(kinematics_time, label_time, self.label_array[:,j])
        self.interpolated_labels = new_label_array
        return 0
