import tqdm
import utils
import numpy as np
from pathlib import Path
from dataset import SimulatorDataset3D

from unet3d.loss import MeshLoss
from unet3d.model import UNet3D

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary


if __name__ == '__main__':

    device = torch.device("cuda")
    root = Path("../checkpoints")
    train_path = ''
    val_path = ''

    epoch_to_use = 44
    use_previous_model = False
    validate_each = 5
    
    in_channels = 3
    out_channels = 3
    final_sigmoid = True
    batch_size = 8
    lr = 5.0e-4
    n_epochs = 500
    momentum=0.9
    img_size = [13, 5, 5]
    
    model = UNet3D(in_channels, out_channels, final_sigmoid).to(device)
    summary(model, input_size=(3, img_size[0], img_size[1], img_size[2]))
    loss_fn = MeshLoss()
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
        model = utils.init_net(model)
        
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

