import tqdm
import utils
import numpy as np
from pathlib import Path
from dataset import SimulatorDataset2D

from model2d import UNet
from loss import MeshLoss2D

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

scale = utils.scale.cuda()[:,:,:,:,0]

def correct(mesh,x):
    corrected = mesh.clone()
    x = (x-0.5)*scale
    corrected = mesh + x
    return corrected

if __name__ == '__main__':

    device = torch.device("cuda")
    root = Path("checkpoints")
    num_workers = 4
    train_set = ['data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9']#, 'data10', 'data11']
    val_set = ['data0']
    # Testing on data1 and data2
    
    path = '../../dataset/2019-10-09-GelPhantom1'
    train_kinematics_path = []
    train_simulator_path = []
    train_label_path = []
    for v in train_set:
        train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
        train_simulator_path = train_simulator_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']
        train_label_path = train_label_path + [path+'/camera/' + v + '_filtered/']

    print(train_kinematics_path)
    val_kinematics_path = []
    val_simulator_path = []
    val_label_path = []
    for v in val_set:
        val_kinematics_path = val_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
        val_simulator_path = val_simulator_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']
        val_label_path = val_label_path + [path+'/camera/' + v + '_filtered/']

    epoch_to_use = 23
    use_previous_model = False
    validate_each = 10
    
    in_channels = 3
    out_channels = 3
    batch_size = 128
    lr = 1.0e-5
    n_epochs = 500
    momentum=0.9

    train_dataset = SimulatorDataset2D(train_kinematics_path, train_simulator_path, train_label_path)
    val_dataset = SimulatorDataset2D(val_kinematics_path, val_simulator_path, val_label_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
#    model = utils.init_net(model)

    loss_fn = MeshLoss2D(batch_size, device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) 
    scheduler = ReduceLROnPlateau(optimizer)

    try:
        model_root = root / "models2d_correction"
        model_root.mkdir(mode=0o777, parents=False)
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

            for i, (kinematics, mesh, label) in enumerate(train_loader):
                kinematics, mesh, label = kinematics.to(device), mesh.to(device), label.to(device)
                pred = model(kinematics, mesh)
                corrected = correct(mesh, pred)
                loss = loss_fn(corrected, label)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss = np.mean(loss.item())
                tq.update(batch_size)
                tq.set_postfix(loss=' loss={:.5f}'.format(mean_loss))
                step += 1
                
            if e % validate_each == 0:
                torch.cuda.empty_cache()
                all_val_loss = []
                with torch.no_grad():
                    model.eval()
                    for j, (kinematics, mesh, label) in enumerate(val_loader):
                        kinematics, mesh, label = kinematics.to(device), mesh.to(device), label
                        pred = model(kinematics, mesh)
                        corrected = correct(mesh, pred)
                        loss = loss_fn(corrected, label)
                        all_val_loss.append(loss.item())
                        
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
