from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import argparse
import os.path as osp

from torch_geometric.nn import GraphNorm
from torch_geometric.transforms import RandomNodeSplit
import torch_geometric.nn.dense.linear as linear


class NetworkInNetwork(torch.nn.Module):
    '''
    Based off GLASSConv, but no mixing and simple 2 linear layers with norm and dropout in between.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ELU(inplace=True),
                 aggr="mean",
                 dropout=0.2):
        super().__init__()
        self.linear1 = linear.Linear(in_channels, out_channels, weight_initializer='glorot')
        self.linear2 = linear.Linear(in_channels + out_channels, out_channels, weight_initializer='glorot')
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x1, edge_index):
        if self.adj.shape[0] == 0:
            n_node = x1.shape[0]
            edge_weight = torch.ones(edge_index[0].shape[0])
            self.adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n_node, n_node))
        x2 = self.activation(self.linear1(x1))
        x2 = self.adj @ x2
        x2 = self.gn(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = torch.cat((x2, x1), dim=-1)
        x2 = self.linear2(x2)
        return x2


def microf1(pred, label):
    '''
    multi-class micro-f1
    '''
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


def run_node_classification(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data')
    dataset = Planetoid(root=path, name=args.dataset)
    data = dataset[0]
    split = RandomNodeSplit(num_val=.10, num_test=.20)(data)
    model = NetworkInNetwork(in_channels=data.x.shape[1], out_channels=7)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")
        print(f"Training loss: {train(optimizer, model, data, split['train_mask'], loss_fn)}")
        print(f"Validation score: {test(model, data, split['val_mask'])}")
        print(f"Testing Score: {test(model, data, split['test_mask'])}")


def train(optimizer, model, dataset, train_mask, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    optimizer.zero_grad()
    pred = model(dataset.x, dataset.edge_index)
    loss = loss_fn(pred[train_mask], dataset.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, dataset, test_mask):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = model(dataset.x, dataset.edge_index)
    return microf1(preds[test_mask].numpy(), dataset.y[test_mask].numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    run_node_classification(args)