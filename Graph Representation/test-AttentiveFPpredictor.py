#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:09:00 2021

@author: elham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import dgl
import torch
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer
from dgllife.utils import mol_to_complete_graph, mol_to_bigraph
from dgllife.model import AttentiveFPPredictor
from torch import nn
 
from torch.utils.data import DataLoader



df = pd.read_csv('reddb_reaction_train.csv')
#df =pd.read_csv('reddb-smiles.csv')
train = df.head(1000)
train_smiles = train['reactantSmiles']
train_y = torch.tensor(train['reactionEnergy']).reshape(-1,1).float()

if torch.cuda.is_available():
    print('use GPU')
    device='cuda'
else:
    print('use CPU')
    device='cpu'

mols = [Chem.MolFromSmiles(s) for s in train_smiles ]

atom_featurizer = CanonicalAtomFeaturizer(atom_data_field = 'h')
n_feats = atom_featurizer.feat_size('h')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='h')
b_feat = bond_featurizer.feat_size('h')

train_graph =[mol_to_bigraph(mol,node_featurizer=atom_featurizer, 
                           edge_featurizer=bond_featurizer) for mol in mols]


model = AttentiveFPPredictor(node_feat_size=n_feats,
                                   edge_feat_size=b_feat,
                                   num_layers=2,
                                   num_timesteps=2,
                                   graph_feat_size=200,
                                   n_tasks=1,
                                   dropout=0.2)
#model = AttentiveFPGNN(n_feats,b_feat,2,200)
model = model.to(device)
print(model)


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))
 
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
     
    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


train_loader = DataLoader(dataset=list(zip(train_smiles, train_graph, train_y)), batch_size=128, collate_fn=collate_molgraphs)


def run_a_train_epoch(n_epochs, epoch, model, data_loader,loss_criterion, optimizer):
    model.train()
    total_loss = 0
    losses = []
     
    for batch_id, batch_data in enumerate(data_loader):
        batch_data
        smiles, bg, labels, masks = batch_data
        if torch.cuda.is_available():
            print('use GPU')
            device='cuda'
        else:
            print('use CPU')
            device='cpu'
            bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
         
        prediction = model(bg, bg.ndata['h'], bg.edata['h'])
        loss = (loss_criterion(prediction, labels)*(masks != 0).float()).mean()
        #loss = loss_criterion(prediction, labels)
        #print(loss.shape)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        losses.append(loss.data.item())
         
    #total_score = np.mean(train_meter.compute_metric('rmse'))
    total_score = np.mean(losses)
    print('epoch {:d}/{:d}, training {:.4f}'.format( epoch + 1, n_epochs,  total_score))
    return total_score

loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0),)
n_epochs = 10
epochs = []
scores = []
for e in range(n_epochs):
    score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
    epochs.append(e)
    scores.append(score)
model.eval()

plt.plot(epochs, scores)


df_test = pd.read_csv('reddb_reaction_test.csv')
#df =pd.read_csv('reddb-smiles.csv')
test = df_test.head(500)
test_smiles = test['reactantSmiles']
test_y = torch.tensor(test['reactionEnergy']).reshape(-1,1).float()


test_mols = [Chem.MolFromSmiles(s) for s in test_smiles ]

test_graph =[mol_to_bigraph(mol,
                           node_featurizer=atom_featurizer, 
                           edge_featurizer=bond_featurizer) for mol in test_mols]

test_loader = DataLoader(dataset=list(zip(test_smiles, test_graph, test_y)), batch_size=128, collate_fn=collate_molgraphs,drop_last=True)


all_pred = []
for batch_id, batch_data in enumerate(test_loader):
    batch_data
    smiles, bg, labels, masks = batch_data
    if torch.cuda.is_available():
        print('use GPU')
        device='cuda'
    else:
        print('use CPU')
        device='cpu'
        bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
         
    pred = model(bg, bg.ndata['h'], bg.edata['h'])
    all_pred.append(pred.data.cpu().numpy())
    

res=model(bg,bg.ndata['h'], bg.edata['h']).detach().numpy()
res = np.vstack(all_pred)
print(res.shape)

plt.clf()
plt.scatter(res,test_y[:384])


