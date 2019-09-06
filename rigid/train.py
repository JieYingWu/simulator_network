import tqdm
import utils
import torch
import numpy as np
from torch import nn
from torch import optim
from pathlib import Path
from loss import RigidLoss
from network import SpringNetwork
from dataset import SimulatorDataset
from torch.utils.data import DataLoader

if __name__=='__main__':

    # Set some parameters
    device = torch.device('cuda')
    root = Path('checkpoints')
    train_path = ['../../dataset/2019-08-08-Lego/data0', '../../dataset/2019-08-08-Lego/data1', '../../dataset/2019-08-08-Lego/data2', '../../dataset/2019-08-08-Lego/data3', '../../dataset/2019-08-08-Lego/data_long']
    val_path = ['../../dataset/2019-08-08-Lego/data4']
    lr = 1e-10
    momentum = 0.9
    batch_size = 8
    num_workers = 8
    n_epochs = 500
    validate_each = 5
    # Input info: 7 coordinates of the tabletoe = 7 + 7 pos from dVRK
    input_size = 14
    output_size = 7
    loss_weight = 10
    
    train_dataset = SimulatorDataset(train_path)
    val_dataset = SimulatorDataset(val_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # If loading from previous model
    use_previous_model = False
    epoch_to_use = 0

    model = SpringNetwork(input_size=input_size, output_size=output_size)
    loss_fn = RigidLoss(loss_weight)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Make directories for saving models and results
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

    # Load previous model if requested
    if use_previous_model:
        path = model_root / 'model_{}.pt'.format(epoch_to_use)
        if path.exists():
            state = torch.load(str(path))
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
        model = utils.init_net(model)

    # List things to save
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

            for i, (input_data, label_data) in enumerate(train_loader):
                input_data, label_data = input_data.to(device), label_data.to(device)
                pred  = model(input_data)
                loss = loss_fn(pred, label_data)
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
                counter=  0
                with torch.no_grad():
                    model.eval()
                    for j, (input_data, label_data) in enumerate(val_loader):
                        input_data, label_data = input_data.to(device), label_data.to(device)
                        pred = model(input_data)
                        val_loss = loss_fn(pred, label_data)
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

