from acorn_pennylane import GraphDataset, InteractionGNN
from qgnn_infer_utils import predict_step
import sys
import yaml 
from torch_geometric.loader import DataLoader
import time
import torch
import yappi

with open(sys.argv[1], "r") as stream:
    hparams = (yaml.load(stream, Loader=yaml.FullLoader))

### model_path should be .pt or .pth file
### scored_graphs_path is folder to store scored graphs in
model_path = sys.argv[2]
scored_graph_path = sys.argv[3]
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device, '\n')

### loading data and model
test_set = GraphDataset(input_dir = '../module_map/testset', hparams = hparams)
print(test_set[0])

test_loader = DataLoader(test_set, batch_size = 1, num_workers= 0)

model = InteractionGNN(hparams,qnn=True).to(device)
model.load_state_dict(torch.load(f'{model_path}'))
print(model)
model.eval()

### score and save each test event
for i, batch in enumerate(test_loader):
    batch = batch.to(device)
    predict_step(model,batch,test_loader, sys.argv[3])
    print(f'test graph {i+1} scored')
    



