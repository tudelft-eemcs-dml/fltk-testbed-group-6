import torch.nn as nn
import torch


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


if __name__ == '__main__':
    model = LRModel()
    params = model.state_dict()
    print(params)
    weight = params['linear.weight']
    print(weight.shape)
    bias = params['linear.bias']
    print(bias.shape)
    print(torch.cat((weight, bias)).shape)

    params = torch.cat((weight, bias))

    s = params > params
    print(s)

    result = torch.where(s, 1, -1)
    print(result)