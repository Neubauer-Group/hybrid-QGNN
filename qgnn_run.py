from acorn_pennylane import GraphDataset, InteractionGNN, loss_function
import sys
import yaml 
from torch_geometric.loader import DataLoader
import time
import torch
import pandas as pd


with open(sys.argv[1], "r") as stream:
    hparams = (yaml.load(stream, Loader=yaml.FullLoader))
print(hparams,'\n')

### model name will be used in file name of saved models and loss log
model_name = sys.argv[2]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device, '\n')

### loading data and creating model
train_set = GraphDataset(input_dir = '../module_map/trainset',hparams = hparams)
val_set = GraphDataset(input_dir = '../module_map/valset', hparams = hparams)
test_set = GraphDataset(input_dir = '../module_map/testset', hparams = hparams)


train_loader = DataLoader(train_set, batch_size = 1, num_workers= 0,shuffle=True)
val_loader = DataLoader(val_set, batch_size = 1, num_workers= 0)
test_loader = DataLoader(test_set, batch_size = 1, num_workers= 0)

print(train_set[0],'\n')

model = InteractionGNN(hparams, qnn = True).to(device)

print(model,'\n')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

tots = time.time()
t_loss=[]
v_loss=[]

### begin training, timing each step
for epoch in range(50):
    torch.save(model.state_dict(), f'saved_models/epoch{epoch}_{model_name}.pth')
    s_epoch = time.time()
    
    model.train(True)
    running_tloss = 0
    for i, tdata in enumerate(train_loader):
        tdata = tdata.to(device)
        optimizer.zero_grad(set_to_none = True)

        fst = time.time()
        out = model(tdata)
        fet = time.time()

        bst = time.time()
        loss, positive_loss, negative_loss = loss_function(out,tdata)
        running_tloss += loss.item()
        loss.backward()
        bet = time.time()

        optimizer.step()

        
        ### printing stats every 10 events
        if i % 9 == 0: 
            results = (f'epoch {epoch}, graph {i+1}, loss {loss.item()}, forward time {fet-fst}, back time {bet-bst}, \n')
            print(results)
        if i ==19: 
            break

    epoch_loss = running_tloss/20

    ### validate once per epoch
    model.eval()
    running_vloss = 0
    with torch.no_grad():
        s = time.time()
        for i, vdata in enumerate(val_loader):
            vdata = vdata.to(device)
            voutputs = model(vdata)
            vloss = loss_function(voutputs,vdata)[0]
            running_vloss += vloss
        val_loss = running_vloss.item()/len(val_loader)
        e = time.time()
    print('validation time', e-s, 'val loss', val_loss)
    e_epoch = time.time()
    print('epoch time', e_epoch-s_epoch, 's')
    
    t_loss.append(epoch_loss)
    v_loss.append(val_loss)


    

model.eval()
running_testloss = 0
with torch.no_grad():
    s = time.time()
    for i, tdata in enumerate(test_loader):
        tdata = tdata.to(device)
        toutputs = model(tdata)
        tloss = loss_function(toutputs,tdata)[0]
        running_testloss += tloss
    test_loss = running_testloss.item()/10
    e = time.time()
print('test time', e-s, 'test loss', test_loss)


tote = time.time()

### save loss to log
df = pd.DataFrame({'train loss': t_loss, 'validation loss': v_loss, 'test loss': test_loss, 'total time':tote-tots})
df.to_csv(f'loss_log_{model_name}.csv')

print(f'total time {tote-tots}s')

    
