import os
import tqdm
import torch
import utils
import numpy as np
import torch.nn as nn
from pathlib import Path
from model import GraphUNet
from dataset import MeshGraphDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

root = Path("checkpoints")
train_set = [ 'data2', 'data3', 'data4',  'data6', 'data7', 'data8', 'data9']#, 'data10', 'data11']
val_set = ['data0']
# Testing on data1
epoch_to_use = 190
use_previous_model = False
validate_each = 10
n_epochs = 2000
lr = 1e-6

####### SET MODEL ########

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GraphUNet(9, 64, 3, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
scheduler = ReduceLROnPlateau(optimizer)
loss_fn = nn.MSELoss()

try:
    model_root = root / "models_batch"
    model_root.mkdir(mode=0o777, parents=False)
except OSError:
    print("path exists")

if use_previous_model:
    model_path = model_root / 'model_{}.pt'.format(epoch_to_use)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch'] + 1
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best_mean_error = state['error']
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

path = '../../dvrk_soft_tissue_simulator/dataset/2019-10-09-GelPhantom1'
train_kinematics_path = []
train_label_path = []
train_fem_path = []
    
for v in train_set:
    train_kinematics_path = train_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
    train_fem_path = train_fem_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']
    
batch_size = 128
num_workers = 6
print(train_kinematics_path)
val_kinematics_path = []
val_label_path = []
val_fem_path = []


for v in val_set:
    val_kinematics_path = val_kinematics_path + [path+'/dvrk/' + v + '_robot_cartesian_velocity.csv']
    val_fem_path = val_fem_path + [path+'/simulator/5e3_fine_mesh/' + v + '/']

train_dataset = MeshGraphDataset(train_kinematics_path, train_fem_path)
val_dataset = MeshGraphDataset(val_kinematics_path, val_fem_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
edge_index = utils.make_edges().to(device)

try:    
    for e in range(epoch, n_epochs + 1):
        
        model.train()
        
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(e, lr))
        epoch_loss = 0
#        train_dataset.__set_network__()

        for i, (fem, fem_next) in enumerate(train_loader):
            fem, fem_next = fem.to(device), fem_next.to(device)
            pred = model(fem.x, fem.edge_index, batch=fem.batch)
            corrected = utils.correct(fem.x, pred)
            loss = loss_fn(corrected, fem_next.x)
            epoch_loss += np.mean(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss = np.mean(loss.item())
            tq.update(batch_size)
            tq.set_postfix(loss=' loss={:.5f}'.format(mean_loss))
                
        tq.set_postfix(loss=' loss={:.5f}'.format(epoch_loss/len(train_loader)))

#        model_path = "augmentation_model.pt"
#        save(e, model, model_path, mean_loss, optimizer, scheduler)
                
        if e % validate_each == 0:
            torch.cuda.empty_cache()
            model.eval()
            all_val_loss = []
            
            with torch.no_grad():
                for j, (fem, fem_next) in enumerate(val_loader):
                    fem, fem_next = fem.to(device), fem_next.to(device)
                    pred = model(fem.x, fem.edge_index, batch=fem.batch)
                    corrected = utils.correct(fem.x, pred)
                    loss = loss_fn(corrected, fem_next.x)
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
