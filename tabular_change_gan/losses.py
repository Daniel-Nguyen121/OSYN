import numpy as np
from scipy.spatial import distance
import torch
from torch import nn

loss_L1 = nn.L1Loss(reduction='sum')
loss_MSE = nn.MSELoss(reduction='sum')

### ----- Loss for element ----- ###

# def L1_loss_element(x, y, clf, device):
#     # x: a feature vector of df
#     # y: one hot vector (true label)
#     # loss_element: L1 max = 2
#     y_pred = torch.tensor(clf.predict_proba(x), device=device)
#     loss = loss_L1(y_pred, torch.tensor([y], device=device)).item()
#     return loss

# def MSE_loss_element(x, y, clf):
#     # x: a feature vector of df
#     # y: one hot vector (true label)
#     # loss_element: MSE max = 2
#     y_pred = torch.tensor(clf.predict_proba(x))
#     loss = loss_MSE(y_pred, torch.tensor([y])).item()
#     return loss

# def JS_loss_element(x, y, clf):
#     # x: a feature vector of df
#     # y: one hot vector (true label)
#     # loss_element: JS max = 1
#     y_pred = torch.tensor(clf.predict_proba(x))
#     loss = distance.jensenshannon(y_pred, torch.tensor([y]), axis = 1).item()
#     return loss

def zero_one_loss_element(x, y, clf):
    # x: a feature vector of df
    # y: scalar (true label)
    # loss_element:  0_1 max = 1
    y_pred = clf.predict(x)
    loss = np.abs(y_pred-y).item()
    return loss

def zero_one_loss_set(X, y, clf):
    y_pred = clf.predict(X)
    loss = np.mean(np.abs(y_pred-y))
    return loss

### ----- Loss for set ----- ###

# def L1_loss_set(X, y, clf):
#     y_pred = torch.tensor(clf.predict_proba(X))
#     loss = loss_L1(y_pred, torch.tensor(y)).item()/X.shape[0]
#     return loss

# def MSE_loss_set(X, y, clf):
#     y_pred = torch.tensor(clf.predict_proba(X))
#     loss = loss_MSE(y_pred, torch.tensor(y)).item()/X.shape[0]
#     return loss

# def JS_loss_set(X, y, clf):
#     y_pred = torch.tensor(clf.predict_proba(X))
#     loss = np.mean(distance.jensenshannon(y_pred, torch.tensor(y), axis = 1))
#     return loss

def zero_one_loss_set(X, y, clf):
    y_pred = clf.predict(X)
    loss = np.mean(np.abs(y_pred-y))
    return loss