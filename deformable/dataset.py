import os
import utils
import torch
import random
import plyfile
import numpy as np
from model import SimuNet, UNet3D
from pathlib import Path
from torch.utils.data import Dataset

scale = torch.tensor([5.28, 7.16, 7.86])/4
def add_gaussian_noise(mesh):
    x = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[0]).unsqueeze(0)
    y = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[1]).unsqueeze(0)
    z = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[2]).unsqueeze(0)
    modifier = torch.cat((x,y,z), axis=0)

#    kn_noise = torch.cat((torch.empty(3).normal_(mean=0, std=2), torch.empty(4).normal_(mean=0, std=0.1)))
    return mesh + modifier#, kinematics + kn_noise

def add_model_noise(model, kinematics, mesh):
    mesh_kinematics = utils.concat_mesh_kinematics(mesh, kinematics)
    correction = model(mesh_kinematics).detach()
    corrected = utils.correct_cpu(mesh, correction)
    return corrected
    
class SimulatorDataset3D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path, augment=False, pc_length=20000):
        self.pc_length = pc_length
        self.augment= augment
        self.kinematics_array = None
        for path in kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')))
        self.kinematics_array = torch.from_numpy(self.kinematics_array).float()
                
        self.simulator_array = []
        for path in simulator_path:
            files = sorted(os.listdir(path))
            self.simulator_array = self.simulator_array + [path + x for x in files]

        self.label_array = []
        for path in label_path:
            files = sorted(os.listdir(path))
            self.label_array = self.label_array + [path + x for x in files]


    def __set_network__(self):
        network_path = Path('augmentation_model.pt')
        self.net = SimuNet(in_channels=10, out_channels=3, dropout=0.1)
        # Load previous model if requested
        if network_path.exists():
            state = torch.load(str(network_path))
            self.net.load_state_dict(state['model'])
#            print('Restored model')
        else:
#            print('Failed to restore model for augmentation')
            self.net = False
            
    def __len__(self):
        return len(self.simulator_array)-1

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        simulation = torch.from_numpy(utils.reshape_volume(np.genfromtxt(self.simulator_array[idx]))).float()
        kinematics = self.kinematics_array[idx,1:]
        
        if self.augment and self.net and idx > 0:
            augmentation = random.random()
            if augmentation < 0.33: # Look back one step
                simulation_prev = utils.reshape_volume(np.genfromtxt(self.simulator_array[idx-1]))
                simulation_prev = torch.from_numpy(simulation_prev).float().unsqueeze(0)
                kinematics_prev = self.kinematics_array[idx-1,1:].unsqueeze(0)
                simulation = add_model_noise(self.net, kinematics_prev, simulation_prev)
                simulation = simulation.squeeze(0)
            elif augmentation < 0.67: # Look back two steps
                simulation_prev = utils.reshape_volume(np.genfromtxt(self.simulator_array[idx-2]))
                simulation_prev = torch.from_numpy(simulation_prev).float().unsqueeze(0)
                kinematics_prev = self.kinematics_array[idx-2,1:].unsqueeze(0)
                simulation = add_model_noise(self.net, kinematics_prev, simulation_prev)
                kinematics_prev = self.kinematics_array[idx-1,1:].unsqueeze(0)
                simulation = add_model_noise(self.net, kinematics_prev, simulation_prev)
                simulation = simulation.squeeze(0)
                
        if self.augment and random.random() < 0.5:
            simulation = add_gaussian_noise(simulation)

            
        pc = plyfile.PlyData.read(self.label_array[idx])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        indices = range(pc.shape[0])
        random.shuffle(indices)
        pc = pc[indices[0:self.pc_length], :]
#        pc = self._pad(pc)
        pc = np.transpose(pc, (1,0))
        simulation_next = torch.from_numpy(utils.reshape_volume(np.genfromtxt(self.simulator_array[idx+1]))).float()

        return kinematics, simulation, torch.from_numpy(pc).float(), simulation_next

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
