import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from math import inf
from copy import deepcopy


class LRModel(nn.Module):
    def __init__(self, input_dim=784, n_classes=10):
        super(LRModel, self).__init__()
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, self.n_classes)

    def forward(self, data):
        batch_size = data.shape[0]
        data = data.view(batch_size, -1)
        pred = self.linear(data)
        return pred



def training(lr_model, train_loader, epoch=1, iteration=500, cuda=False, learning_rate=1e-3, log_interval=50):
    train_iter = iter(train_loader)
    best_er = inf
    best_celoss = inf
    best_model_wts_er = deepcopy(lr_model.state_dict())
    best_model_wts_loss = deepcopy(lr_model.state_dict())
    for e in range(epoch):
        total_incorrect = 0
        total = 0
        total_celoss = 0

        for i in range(1, iteration + 1):
            lr_model.train()
            optimizer = torch.optim.Adam([{'params': lr_model.linear.parameters(), 'lr': learning_rate},])

            try:
                data, label = train_iter.next()
            except Exception as error:
                train_iter = iter(train_loader)
                data, label = train_iter.next()

            if cuda:
                data, label = data.cuda(), label.cuda()

            data, label = Variable(data.view(-1, 28 * 28)), Variable(label)
            optimizer.zero_grad()

            incorrect, er, ce_loss = lr_model(data, label)
            total_incorrect += incorrect
            total += label.shape[0]
            total_celoss += ce_loss
            ce_loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print('Epoch: {}/{} Train iter: {} [({:.0f}%)]\tCE_Loss: {:.6f}\tError Rate: {:.6f}'.format(
                    e+1, epoch, i, 100. * i / iteration, ce_loss.item(), er))

        if best_er > total_incorrect/total:
            best_er = total_incorrect/total
            best_model_wts_er = deepcopy(lr_model.state_dict())
        if best_celoss > total_celoss/iteration:
            best_celoss = total_celoss/iteration
            best_model_wts_loss = deepcopy(lr_model.state_dict())

    best_model_er = lr_model.load_state_dict(best_model_wts_er)
    best_model_loss = lr_model.load_state_dict(best_model_wts_loss)

    return best_model_er, best_model_loss
