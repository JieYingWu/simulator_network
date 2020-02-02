import tqdm
import utils
import numpy as np
from pathlib import Path
from dataset import SimulatorDataset3D

from model import UNet3D, SimuNet
from loss import MeshLoss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FEM_WEIGHT = 1000
 
if __name__ == '__main__':
    
    root = Path("checkpoints")
    num_workers = 4
    train_set = ['data6']# 'data3', 'data4', 'data5',  'data6', 'data7', 'data8', 'data9', 'data10', 'data11']
    val_set = ['data4']# 'data0']
    # Testing on data1 and 2
    
    path = '../../dataset/2019-10-09-GelPhantom1'
    train_kinematics_path = []
    train_simulator_path = []
    train_label_path = []
    for v in train_set:
        train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_processed_interpolated.csv']
        train_simulator_path = train_simulator_path + [path+'/simulator/5e3_data/' + v + '/']
        train_label_path = train_label_path + [path+'/camera/' + v + '_filtered/']
#    for v in train_set:
#        train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_processed_interpolated.csv']
#        train_simulator_path = train_simulator_path + [path+'/simulator/5e3_data/' + v + '_net/']
#        train_label_path = train_label_path + [path+'/camera/' + v + '_filtered/']


    print(train_kinematics_path)
    val_kinematics_path = []
    val_simulator_path = []
    val_label_path = []
    for v in val_set:
        val_kinematics_path = val_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_processed_interpolated.csv']
        val_simulator_path = val_simulator_path + [path+'/simulator/5e3_data/' + v + '/']
        val_label_path = val_label_path + [path+'/camera/' + v + '_filtered/']

    epoch_to_use = 7
    use_previous_model = False
    validate_each = 5
    
    in_channels = 10
    out_channels = 3
    batch_size = 128
    lr = 1.0e-7
    n_epochs = 500
    momentum=0.9

    train_dataset = SimulatorDataset3D(train_kinematics_path, train_simulator_path, train_label_path, augment=True)
    val_dataset = SimulatorDataset3D(val_kinematics_path, val_simulator_path, val_label_path, augment=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = SimuNet(in_channels=in_channels, out_channels=out_channels, dropout=0.1).to(device)
#    model = utils.init_net(model)
#    summary(model, input_size=(3, img_size[0], img_size[1], img_size[2]))

    base_mesh = np.genfromtxt('../../dataset/2019-10-09-GelPhantom1/simulator/5e3_data/data0/position0001.txt')
    base_mesh = torch.from_numpy(base_mesh).float().to(device)
    base_mesh = base_mesh.reshape(13,5,5,3).permute(3,0,1,2).unsqueeze(0)
    print(base_mesh.size())
#    print(base_mesh[0,1,:,-1,:])

    loss_fn = MeshLoss(batch_size, FEM_WEIGHT, base_mesh, device)
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
    
    try:    
        for e in range(epoch, n_epochs + 1):
#            for param_group in optimizer.param_groups:
#                print('Learning rate ', param_group['lr'])
        
            model.train()

            tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
            tq.set_description('Epoch {}, lr {}'.format(e, lr))
            epoch_loss = 0
            train_dataset.__set_network__()

            for i, (kinematics, mesh, label, mesh_next) in enumerate(train_loader):
                kinematics, mesh, label, mesh_next = kinematics.to(device), mesh.to(device), label.to(device), mesh_next.to(device)

                mesh_kinematics = utils.concat_mesh_kinematics(mesh, kinematics)
                pred = model(mesh_kinematics)
                corrected = utils.correct(mesh, pred)
                loss = loss_fn(corrected, label, mesh_next)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss = np.mean(loss.item())
                tq.update(batch_size)
                tq.set_postfix(loss=' loss={:.5f}'.format(mean_loss))
                step += 1

            model_path = "augmentation_model.pt"
            save(e, model, model_path, mean_loss, optimizer, scheduler)

            if e % validate_each == 0:
                torch.cuda.empty_cache()
                all_val_loss = []
                with torch.no_grad():
#                    model.eval()
                    for j, (kinematics, mesh, label, mesh_next) in enumerate(val_loader):
                        kinematics, mesh, label, mesh_next = kinematics.to(device), mesh.to(device), label.to(device), mesh_next.to(device)
                        mesh_kinematics = utils.concat_mesh_kinematics(mesh, kinematics)
                        pred = model(mesh_kinematics)
                        corrected = utils.correct(mesh, pred)
                        loss = loss_fn(corrected, label, mesh_next)
                        all_val_loss.append(loss.item())

                        if (j == 500):
                            mesh_plot = mesh.detach().cpu().numpy()
                            pred_plot = pred.detach().cpu().numpy()
                            label_plot = label.detach().cpu().numpy()
                            np.save(str(results_root/'prediction_{e}'.format(e=e)), pred_plot)
                            np.save(str(results_root/'mesh_{e}'.format(e=e)), mesh_plot)
                            np.save(str(results_root/'label_{e}'.format(e=e)), label_plot)

                        
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

