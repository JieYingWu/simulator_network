import os
import utils
import torch
import random
import plyfile
import numpy as np
from model import UNet3D, SimuAttentionNet
from pathlib import Path
from torch.utils.data import Dataset

augment_steps = 1
scale = utils.scale_cpu[0,:,0,0,0].numpy()/32

def add_gaussian_noise(mesh):
    x = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[0]).unsqueeze(0)
    y = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[1]).unsqueeze(0)
    z = torch.empty(mesh.size()[1:]).normal_(mean=0,std=scale[2]).unsqueeze(0)
    modifier = torch.cat((x,y,z), axis=0)

#    kn_noise = torch.cat((torch.empty(3).normal_(mean=0, std=2), torch.empty(4).normal_(mean=0, std=0.1)))
    return mesh + modifier#, kinematics + kn_noise

class SimulatorDataset(Dataset):
    def __init__(self, kinematics_path, fem_path, augment=False):
        self.augment= augment
        self.kinematics_array = None
        for path in kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')[:,1:utils.FIELDS+1]
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')[:,1:utils.FIELDS+1]))
        self.kinematics_array = np.concatenate((self.kinematics_array[:,0:3], self.kinematics_array[:,7:10]), axis=1)
        self.kinematics_array = torch.from_numpy(self.kinematics_array).float()
                
        self.fem_array = []
        for path in fem_path:
            files = sorted(os.listdir(path))
            self.fem_array = self.fem_array + [path + x for x in files]

        print(self.kinematics_array.shape, len(self.fem_array))
    def __set_network__(self):
        network_path = Path('augmentation_model.pt')
        self.net = UNet3D(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS)
        # Load previous model if requested
        if network_path.exists():
            state = torch.load(str(network_path))
            self.net.load_state_dict(state['model'])
#            print('Restored model')
        else:
            print('Failed to restore model for augmentation')
            self.net = False
            
    def __len__(self):
        return len(self.fem_array)-1

    def add_model_noise(self, idx, steps):
        if idx < steps:
            idx = steps
            
        simulation_prev = utils.reshape_volume(np.genfromtxt(self.fem_array[idx - steps]))
        simulation_prev = torch.from_numpy(simulation_prev).float().unsqueeze(0)

        for i in range(steps,0,-1):
            kinematics_prev = self.kinematics_array[idx-i].unsqueeze(0)
            correction = self.net(simulation_prev, kinematics_prev).detach()
            simulation_prev = utils.correct_cpu(simulation_prev, correction)+simulation_prev

        return simulation_prev.squeeze(0)
    
    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        kinematics = self.kinematics_array[idx]
        
        if self.augment and random.random() < 0.3:
            fem = utils.reshape_volume(np.genfromtxt(self.fem_array[idx]))
            fem = torch.from_numpy(fem).float()
            fem = add_gaussian_noise(fem)
        elif self.augment and self.net and random.random() < 0.5:
            value = random.randint(0, augment_steps)
            fem = self.add_model_noise(idx, value)
        else:
            fem = utils.reshape_volume(np.genfromtxt(self.fem_array[idx]))
            fem = torch.from_numpy(fem).float()

        fem_next = torch.from_numpy(utils.reshape_volume(np.genfromtxt(self.fem_array[idx+1]))).float() - fem
        print(kinematics)
        print(torch.mean(fem_next[:]))
        return kinematics, fem, fem_next#, pc_last


class SimulatorDatasetPC(SimulatorDataset):
    def __init__(self, kinematics_path, label_path, fem_path, augment=False, pc_length=26000):
        super(SimulatorDatasetPC, self).__init__(kinematics_path, label_path, fem_path, augment=False)
        self.pc_length = pc_length

        self.label_array = []
        for path in label_path:
            files = sorted(os.listdir(path))
            self.label_array = self.label_array + [path + x for x in files]
                
    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
        kinematics = self.kinematics_array[idx]
        
        kinematics, simulation, fem = super(SimulatorDatasetPC, self).__getitem__(idx)

        # Current point cloud
        # pc_last = plyfile.PlyData.read(self.label_array[idx])['vertex']
        # pc_last = np.concatenate((np.expand_dims(pc_last['x'], 1), np.expand_dims(pc_last['y'],1), np.expand_dims(pc_last['z'],1)), 1)
        # indices = range(pc_last.shape[0])
        # random.shuffle(indices)
        # pc_last = pc_last[indices[0:self.pc_length], :]
        # pc_last = torch.from_numpy(np.transpose(pc_last, (1,0))).float()

        # Next point cloud
        pc = plyfile.PlyData.read(self.label_array[idx+1])['vertex']
        pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
        indices = range(pc.shape[0])
        random.shuffle(indices)
        pc = pc[indices[0:self.pc_length], :]
        pc = torch.from_numpy(np.transpose(pc, (1,0))).float()

        return kinematics, simulation, pc, fem#, pc_last
    
class SimulatorDataset2D(Dataset):
    def __init__(self, kinematics_path, simulator_path, label_path, augment=False, pc_length=24000):
        self.pc_length = pc_length
        self.augment = augment
        self.kinematics_array = None
        for path in  kinematics_path:
            if self.kinematics_array is None:
                self.kinematics_array = np.genfromtxt(path, delimiter=',')
            else:
                self.kinematics_array = np.concatenate((self.kinematics_array, np.genfromtxt(path, delimiter=',')))
        self.kinematics_array = torch.from_numpy(self.kinematics_array[:,1:]).float()

#        self.simulator_array = []
        self.all_fem = None
        for path in simulator_path:
            print(path)
            fem = np.load(path[0:-1]+'.npy')
            fem = self._reshape(fem)
            fem = torch.from_numpy(fem).float()
            if self.all_fem is None:
                self.all_fem = fem
            else:
                self.all_fem = torch.cat((self.all_fem, fem), axis=0)
#            files = sorted(os.listdir(path))
#            self.simulator_array = self.simulator_array + [path + x for x in files]
#            for x in files:
#                fem = np.genfromtxt(path+x)
#                fem = torch.from_numpy(fem).float()
#                fem = fem.unsqueeze(0)
#                if self.all_fem is None:
#                    self.all_fem = fem
#                else:
#                    self.all_fem = torch.cat((self.all_fem, fem), axis=0)

#        self.label_array = []

        self.all_pc = None
        for path in label_path:
#            files = sorted(os.listdir(path))
            print(path)
            pc = np.load(path[0:-1]+'.npy')
            pc = torch.from_numpy(pc).float()
#            pc = pc[0:-1,:,:]
            if self.all_pc is None:
                self.all_pc = pc
            else:
                self.all_pc = torch.cat((self.all_pc, pc), axis=0)

#            self.label_array = self.label_array + [path + x for x in files]
#            for x in files:
#                pc = plyfile.PlyData.read(path+x)['vertex']
#                pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
#                pc = self._truncate(pc)
#                pc = torch.from_numpy(pc).float()
#                pc = pc.unsqueeze(0)
#                if self.all_pc is None:
#                    self.all_pc = pc
#                else:
#                    self.all_pc = torch.cat((self.all_pc, pc), axis=0)

            
    def __len__(self):
        return self.all_fem.size()[0]
        #return len(self.simulator_array)

    # return robot kinematics, mesh, and point cloud
    def __getitem__(self, idx):
#        simulation_time = time()
#        simulation = np.genfromtxt(self.simulator_array[idx])
#        simulation = torch.from_numpy(self._reshape(simulation)).float()
        simulation = self.all_fem[idx,:,:]
        if self.augment:
            simulation = add_gaussian_noise(simulation)
#        print('Loading simulation took ' + str(time()-simulations_time) + ' s')

#        label_time = time()
#        print('Loading label took ' + str(time()-label_time) + ' s')
#        pc = plyfile.PlyData.read(self.label_array[idx])['vertex']
 #       pc = np.concatenate((np.expand_dims(pc['x'], 1), np.expand_dims(pc['y'],1), np.expand_dims(pc['z'],1)), 1)
 #       pc = self._truncate(pc)
 #       pc = torch.from_numpy(pc).float()

        pc = self.all_pc[idx,:,:]
         
        return self.kinematics_array[idx,:], simulation, pc

    def _reshape(self, x):
        y = x.reshape(-1, utils.VOL_SIZE[0], utils.VOL_SIZE[1], utils.VOL_SIZE[2], utils.VOL_SIZE[3])
        y = y.transpose((0, 4, 1, 2, 3))
        y = y[:,:,:,-1,:]
        return y

    def _pad(self, x):
        padded = np.zeros((self.pc_length, 3))
        padded[0:x.shape[0],:] = x
        return padded

    def _truncate(self, x):
        indices = torch.randperm(self.pc_length)
        truncated = x[indices, :]
        return truncated
