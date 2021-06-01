import os
import tqdm
import torch
import utils
import numpy as np
import torch.nn as nn
from pathlib import Path
from model import GraphUNet, BSplineNet
from dataset import MeshGraphDataset
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

root = Path("checkpoints")
train_set = ['data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9']#, 'data10', 'data11']
val_set = ['data0']
# Testing on data1
epoch_to_use = 50 # Made step size bigger at 470, Added augmentatio in 700
use_previous_model = True
validate_each = 10
n_epochs = 5000
lr = 1e-4
batch_size = 16
momentum = 0.9
augment = True

####### SET MODEL ########

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GraphUNet(utils.IN_CHANNELS, 256, 3, utils.NET_DEPTH, sum_res=False).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = ReduceLROnPlateau(optimizer)
loss_fn = nn.MSELoss()

try:
    model_root = root / "UNet" / "fine_mesh_half_step"
    model_root.mkdir(mode=0o777, parents=False)
except OSError:
    print("path exists")

if use_previous_model:
    model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path), map_location=device)
        epoch = state['epoch'] + 1
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_mean_error = state['error']
        del state
        print('Restored model, epoch {}'.format(epoch))
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
    'error': error,
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
}, str(model_path))

####### SET TRAIN AND VAL SETS ########

path = '../../dataset/2019-10-09-GelPhantom1'
train_kinematics_path = []
train_label_path = []
train_fem_path = []
    
for v in train_set:
    train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
    train_fem_path = train_fem_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']
    
print(train_kinematics_path)
val_kinematics_path = []
val_label_path = []
val_fem_path = []


for v in val_set:
    val_kinematics_path = val_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
    val_fem_path = val_fem_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']

train_dataset = MeshGraphDataset(train_kinematics_path, train_fem_path, augment=augment)
val_dataset = MeshGraphDataset(val_kinematics_path, val_fem_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

try:    
    for e in range(epoch, n_epochs + 1):
        
        model.train()
        if augment:
            train_dataset.__set_network__()
        
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, lr))
        epoch_loss = 0
        
        for i, (fem, corr) in enumerate(train_loader):
            fem, corr = fem.to(device), corr.to(device)
            pred = model(fem.x, fem.edge_index, batch=fem.batch)
            corrected = utils.correct(fem.x, pred, device)
            loss = loss_fn(corrected, corr.x)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(loss.item()))

        mean_loss = epoch_loss/len(train_loader)
        tq.set_postfix(loss=' loss={:.5f}'.format(mean_loss))

#        model_path = "augmentation_model.pt"
#        save(e, model, model_path, mean_loss, optimizer, scheduler)
        
        if e % validate_each == 0:
            torch.cuda.empty_cache()
            model.eval()
            mean_loss = 0
            
            with torch.no_grad():
                for j, (fem, corr) in enumerate(val_loader):
                    fem, corr = fem.to(device), corr.to(device)
                    pred = model(fem.x, fem.edge_index, batch=fem.batch)
                    corrected = utils.correct(fem.x, pred, device)
                    loss = loss_fn(corrected, corr.x)
                    mean_loss += loss.item()

            mean_loss = mean_loss / len(val_loader)
            tq.set_postfix(loss='validation loss={:5f}'.format(mean_loss))
            scheduler.step(mean_loss)

            best_mean_rec_loss = mean_loss
            model_path = model_root / "model_{}.pt".format(e)
            save(e, model, model_path, best_mean_rec_loss, optimizer, scheduler)

            model_path = "augmentation_model.pt"
            save(e, model, model_path, mean_loss, optimizer, scheduler)

                    
        tq.close()

except KeyboardInterrupt:
    tq.close()
    print('Ctrl+C, done.')
    exit()
