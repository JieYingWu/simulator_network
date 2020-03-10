
import tqdm
import utils
import numpy as np
from pathlib import Path
from dataset import SimulatorDataset3D

from model import UNet3D, SimuNet, SimuNetWithSurface, SimuAttentionNet
from loss import MeshLoss

import sys
sys.path.insert(0,'../simulator/')
from play_simulation import play_simulation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEM_WEIGHT = 1000
REG_WEIGHT = 0.0001

if __name__ == '__main__':
    
    root = Path("checkpoints")
    num_workers = 6
    train_set = [ 'data4', 'data5',  'data6', 'data7', 'data8', 'data9', 'data10', 'data11']
    val_set = ['data0']
    # Testing on data1 and 2
    
    path = '../../dataset/2019-10-09-GelPhantom1'
    train_kinematics_path = []
    train_simulator_path = []
    train_label_path = []
    train_fem_path = []
    
    for v in train_set:
        train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
        train_simulator_path = train_simulator_path + [v + '/']
        train_label_path = train_label_path + [path+'/camera/' + v + '_filtered/']
        train_fem_path = train_fem_path + [path+'/simulator/5e3_data/' + v + '/']

    print(train_kinematics_path)
    val_kinematics_path = []
    val_simulator_path = []
    val_label_path = []
    val_fem_path = []

    for v in val_set:
        val_kinematics_path = val_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
        val_simulator_path = val_simulator_path + [v + '/']
        val_label_path = val_label_path + [path+'/camera/' + v + '_filtered/']
        val_fem_path = val_fem_path + [path+'/simulator/5e3_data/' + v + '/']

    epoch_to_use = 75
    use_previous_model = True
    validate_each = 5
    play_each = 2000
    
    batch_size = 32
    lr = 1.0e-6
    n_epochs = 1000
    momentum=0.9

    base_mesh = np.genfromtxt('../simulator/mesh.txt')
    base_mesh = torch.from_numpy(base_mesh).float()
    base_mesh = base_mesh.reshape(utils.VOL_SIZE).permute(3,0,1,2).unsqueeze(0)

    train_dataset = SimulatorDataset3D(train_kinematics_path, train_simulator_path, train_label_path, train_fem_path, augment=True)
    val_dataset = SimulatorDataset3D(val_kinematics_path, val_simulator_path, val_label_path, val_fem_path, augment=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = SimuAttentionNet(in_channels=utils.IN_CHANNELS, out_channels=utils.OUT_CHANNELS, dropout=utils.DROPOUT).to(device)
    if not use_previous_model:
        model = utils.init_net(model)
#    summary(model, input_size=(3, img_size[0], img_size[1], img_size[2]))

    base_mesh = base_mesh.to(device)
    loss_fn = MeshLoss(FEM_WEIGHT, REG_WEIGHT,  base_mesh, device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) 
    scheduler = ReduceLROnPlateau(optimizer)
    
    try:
        model_root = root / "models"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    try:
        results_root = root / "results"
        results_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")
    
    # Read existing weights for both G and D models
    if use_previous_model:
        model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
        if model_path.exists():
            state = torch.load(str(model_path))
            epoch = state['epoch'] + 1
            step = state['step']
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            best_mean_error = state['error']
            print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        else:
            print('Failed to restore model')
            exit()
    else:
        epoch = 1
        step = 0
        best_mean_error = 0.0
        
    save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'error': error,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, str(model_path))

    print('In train_unet.py, using 3 losses: Chamfer, FEM L2 and regularization')
    
    try:    
        for e in range(epoch, n_epochs + 1):
        
            model.train()

            tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
            tq.set_description('Epoch {}, lr {}'.format(e, lr))
            epoch_loss = 0
            train_dataset.__set_network__()

            for i, (kinematics, mesh, pc, fem) in enumerate(train_loader):
                kinematics, mesh, pc, fem = kinematics.to(device), mesh.to(device), pc.to(device), fem.to(device)
#            for i, (kinematics, mesh, pc, fem, pc_last) in enumerate(train_loader):
#                kinematics, mesh, pc, fem, pc_last = kinematics.to(device), mesh.to(device), pc.to(device), fem.to(device), pc_last.to(device)
#                mesh_kinematics = utils.concat_mesh_kinematics(mesh, kinematics)
                pred = model(mesh, kinematics)
                corrected = utils.correct(mesh, pred)
                loss = loss_fn(corrected, pc, fem, pred)
                epoch_loss += np.mean(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss = np.mean(loss.item())
                tq.update(batch_size)
                tq.set_postfix(loss=' loss={:.5f}'.format(mean_loss))
                step += 1
                
            tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(train_loader)))

            model_path = "augmentation_model.pt"
            save(e, model, model_path, mean_loss, optimizer, scheduler)
                
            if e % play_each == 0:
                for idx in range(len(train_set)):
                    kinematics = torch.from_numpy(np.genfromtxt(train_kinematics_path[idx], delimiter=',')).float().to(device)
                    play_simulation(model, base_mesh, kinematics, train_set[idx])
            
            if e % validate_each == 0:
                torch.cuda.empty_cache()
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                all_val_loss = []
                with torch.no_grad():
                    for j, (kinematics, mesh, pc, fem) in enumerate(val_loader):
                        kinematics, mesh, pc, fem = kinematics.to(device), mesh.to(device), pc.to(device), fem.to(device)
#                    for j, (kinematics, mesh, pc, fem, pc_last) in enumerate(val_loader):
#                        kinematics, mesh, pc, fem, pc_last = kinematics.to(device), mesh.to(device), pc.to(device), fem.to(device), pc_last.to(device)
#                        mesh_kinematics = utils.concat_mesh_kinematics(mesh, kinematics)
                        pred = model(mesh, kinematics)
                        corrected = utils.correct(mesh, pred)
                        loss = loss_fn(corrected, pc, fem, pred)
                        all_val_loss.append(loss.item())

                        if (j == 500):
                            mesh_plot = mesh.detach().cpu().numpy()
                            pred_plot = pred.detach().cpu().numpy()
                            label_plot = label.detach().cpu().numpy()
                            np.save(str(results_root/'prediction_{e}'.format(e=e)), pred_plot)
                            np.save(str(results_root/'mesh_{e}'.format(e=e)), mesh_plot)
                            np.save(str(results_root/'label_{e}'.format(e=e)), label_plot)

                    
#                for idx in range(len(val_set)):
#                    kinematics = torch.from_numpy(np.genfromtxt(val_kinematics_path[idx], delimite#r=',')).float().to(device)
#                    play_simulation(model, base_mesh, kinematics, val_set[idx])
            
                mean_loss = np.mean(all_val_loss)
                tq.set_postfix(loss='validation loss={:5f}'.format(mean_loss))
                scheduler.step(mean_loss)

                best_mean_rec_loss = mean_loss
                model_path = model_root / "model_{}.pt".format(e)
                save(e, model, model_path, best_mean_rec_loss, optimizer, scheduler)
            
            tq.close()

    except KeyboardInterrupt:
        tq.close()
        print('Ctrl+C, done.')
        exit()

