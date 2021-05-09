import torch
import torch.nn as nn
import numpy as np


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

    def __init__(self, rule, attack_type='no', device_num=20, compromised=5):
        self.model = Model()
        self.rule = rule
        self.attack_type = attack_type
        self.devices = [Trainer() for _ in range(device_num)]
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
        if self.attack_type == 'flip':
            return
        if self.attack_type == 'no':
            return
        i = 0
        for v in self.model.state_dict().values():
            v = v.flatten()
            for j in range(len(self.states[i][0])):
                result = self._attack([self.states[i][k][j] for k in range(self.device_num)], v[j])
                for k in range(len(result)):
                    self.states[i][k][j] = result[k]
            i += 1

    def _attack(self, params, original):
        res = params
        if self.attack_type == 'partial':
            res = self.poisoning(params, original, False)
        if self.attack_type == 'full':
            res = self.poisoning(params, original, True)
        if self.attack_type == 'gaussian':
            res = self.gaussian(params)
        return res

    def gaussian(self, params):
        params = torch.Tensor(params)
        wmean = torch.mean(params)
        std = torch.std(params)
        for i in range(self.compromised):
            params[i] = np.random.normal(wmean, std)
        return params

    @staticmethod
    def trimmed(params, trim):
        if torch.is_tensor(params):
            params, _ = torch.sort(params)
        else:
            params = sorted(params)
        params = params[trim:-trim]
        res = sum(params) / len(params)
        return float(res)

    def poisoning(self, params, original, full):
        params = torch.Tensor(params)
        if self.rule == 'trimmed' or self.rule == 'medium':
            if full:
                res = self.trimmed(params, self.compromised)
                wmax = torch.max(params[self.compromised:])
                wmin = torch.min(params[self.compromised:])
                if res > original:
                    for i in range(self.compromised):
                        if wmin <= 0:
                            params[i] = np.random.uniform(2 * wmin, wmin)
                        else:
                            params[i] = np.random.uniform(wmin / 2, wmin)
                else:
                    for i in range(self.compromised):
                        if wmax >= 0:
                            params[i] = np.random.uniform(wmax, 2 * wmax)
                        else:
                            params[i] = np.random.uniform(wmax / 2, wmax)
                return params
            else:
                wmean = torch.mean(params[:self.compromised])
                std = torch.std(params[:self.compromised])
                if wmean > original:
                    for i in range(self.compromised):
                        params[i] = np.random.uniform(wmean - 4 * std, wmean - 3 * std)
                else:
                    for i in range(self.compromised):
                        params[i] = np.random.uniform(wmean + 3 * std, wmean + 4 * std)
                return params

    def aggregate(self, params):
        if self.rule == 'trimmed':
            return self.trimmed(params, self.compromised)
        if self.rule == 'medium':
            return self.trimmed(params, (params - 1) // 2)
        return params[0]

    def update(self):
        d = {}
        i = 0
        for k, v in self.model.state_dict().items():
            t = torch.zeros_like(self.states[i][0])
            for j in range(t.shape[0]):
                t[j] = self.aggregate([self.states[i][p][j] for p in range(self.device_num)])
            d[k] = t.reshape_as(v)
            d[k].requires_grad = v.requires_grad
            i += 1
        self.model.load_state_dict(d)
        print(self.states[0][0])
        print(self.model.state_dict())


master = Master('trimmed', 'partial')
master.step()

