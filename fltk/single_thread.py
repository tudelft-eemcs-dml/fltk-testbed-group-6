import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


class Trainer(object):

    def __init__(self):
        self.model = Model()
        self.x = torch.rand((32, 10))
        self.y = torch.randint(high=2, size=(32,))
        self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), 1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.zero_grad()
        loss = self.criterion(self.model(self.x), self.y)
        loss.backward()
        self.optimizer.step()


class Master(object):

    def __init__(self, rule, type='no', device_num=20, compromised=5):
        self.model = Model()
        self.rule = rule
        self.type = type
        self.devices = [Trainer()] * device_num
        self.device_num = device_num
        self.compromised = compromised
        self.states = []

    def step(self):

        for device in self.devices:
            device.model.load_state_dict(self.model.state_dict())
            device.train()
        self.states = []
        for i in range(len(self.devices)):
            self.states.append([])
            for param in self.devices[i].model.parameters():
                self.states[i].append(param.flatten())
        self.states = list(zip(*self.states))
        self.attack()
        self.update()

    def attack(self):
        if self.type == 'flip':
            return
        if self.type == 'no':
            return
        i = 0
        for v in self.model.state_dict().values():
            v = v.flatten()
            for j in range(len(self.states[i][0])):
                result = self._attack([self.states[i][k][j] for k in range(self.device_num)], v[j])
                for k in range(len(result)):
                    self.states[i][k][j] = result[k]
            i += 1

    def _attack(self, x, original):
        return x

    def _poisoning(self, full):
        pass

    def aggregate(self, params):
        return params[0]

    def update(self):
        d = {}
        i = 0
        for k, v in self.model.state_dict().items():
            t = torch.zeros_like(self.states[i][0])
            for j in range(t.shape[0]):
                t[j] = self.aggregate([self.states[i][p][j] for p in range(self.device_num)])
            d[k] = t.reshape_as(v)
            i += 1
        self.model.load_state_dict(d)
        print(self.states[0][0])
        print(self.model.state_dict())


master = Master('trimmed', 'partial')
master.step()

