import os
import utils
import torch
import random
import plyfile
import numpy as np
from model import SimuNet, UNet3D
from pathlib import Path
from torch.utils.data import Dataset

augment_steps = 60.0

scale = torch.tensor([5.28, 7.16, 7.86])/10
def add_gaussian_noise(mesh):
    x = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[0]).unsqueeze(0)
    y = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[1]).unsqueeze(0)
    z = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[2]).unsqueeze(0)
    modifier = torch.cat((x,y,z), axis=0)

#    kn_noise = torch.cat((torch.empty(3).normal_(mean=0, std=2), torch.empty(4).normal_(mean=0, std=0.1)))
    return mesh + modifier#, kinematics + kn_noise
    
class SimulatorDataset3D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path, augment=False, pc_length=27000):
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
        self.net = SimuNet(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS, dropout=utils.DROPOUT)
        # Load previous model if requested
        if network_path.exists():
            state = torch.load(str(network_path))
            self.net.load_state_dict(state['model'])
#            print('Restored model')
        else:
#            print('Failed to restore model for augmentation')
            self.net = False
            
    def __len__(self):
        return 300#len(self.simulator_array)-1

    def add_model_noise(self, idx, steps):
        if idx < steps:
            idx = steps
            
        simulation_prev = utils.reshape_volume(np.genfromtxt(self.simulator_array[idx - steps]))
        simulation_prev = torch.from_numpy(simulation_prev).float().unsqueeze(0)

        for i in range(steps,0,-1):
            kinematics_prev = self.kinematics_array[idx-i,1:1+utils.FIELDS].unsqueeze(0)
            mesh_kinematics = utils.concat_mesh_kinematics(simulation_prev, kinematics_prev)
            correction = self.net(mesh_kinematics).detach()
            simulation_prev = utils.correct_cpu(simulation_prev, correction)

        return simulation_prev.squeeze(0)
    
    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        kinematics = self.kinematics_array[idx,1:1+utils.FIELDS]
        
 #       if self.augment and self.net:
 #           value = random.randint(0, augment_steps)
 #           simulation = self.add_model_noise(idx, value)
 #       else:
        simulation = utils.reshape_volume(np.genfromtxt(self.simulator_array[idx]))
        simulation = torch.from_numpy(simulation).float()

     
        if self.augment and random.random() < 0.8:
            simulation = add_gaussian_noise(simulation)
            
        pc = plyfile.PlyData.read(self.label_array[idx+1])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        indices = range(pc.shape[0])
        random.shuffle(indices)
        pc = pc[indices[0:self.pc_length], :]
        pc = np.transpose(pc, (1,0))
#        simulation_next = torch.from_numpy(utils.reshape_volume(np.genfromtxt(self.simulator_array[idx+1]))).float()

        return kinematics, simulation, torch.from_numpy(pc).float()#, simulation_next

    
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
