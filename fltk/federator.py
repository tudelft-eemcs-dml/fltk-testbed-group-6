import os
import datetime
import time
from typing import List

import torch
import numpy as np

from dataclass_csv import DataclassWriter
from torch.distributed import rpc

from fltk.client import Client
from fltk.datasets.data_distribution import distribute_batches_equally
from fltk.strategy.client_selection import random_selection
from fltk.util.arguments import Arguments
from fltk.util.base_config import BareConfig
from fltk.util.data_loader_utils import load_train_data_loader, load_test_data_loader, \
    generate_data_loaders_from_distributed_dataset
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

from fltk.util.results import EpochData
from fltk.util.tensor_converter import convert_distributed_data_into_numpy

logging.basicConfig(level=logging.DEBUG)


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer

    def __repr__(self):
        return self.name


class Federator:
    """
    Central component of the Federated Learning System: The Federator

    The Federator is in charge of the following tasks:
    - Have a copy of the global model
    - Client selection
    - Aggregating the client model weights/gradients
    - Saving all the metrics
        - Use tensorboard to report metrics
    - Keep track of timing

    """
    clients: List[ClientRef] = []
    epoch_counter = 0
    client_data = {}

    def __init__(self, client_id_triple, num_epochs = 3, config=None):
        log_rref = rpc.RRef(FLLogger())
        self.log_rref = log_rref
        self.num_epoch = num_epochs
        self.config = config
        self.args = config
        self.tb_path = config.output_location
        self.ensure_path_exists(self.tb_path)
        self.tb_writer = SummaryWriter(f'{self.tb_path}/{config.experiment_prefix}_federator')
        self.create_clients(client_id_triple)
        self.config.init_logger(logging)

        self.model = self.load_default_model()
        self.rule = 'trimmed'
        self.attack_type = 'full'
        self.compromised = 2
        self.device_num = self.config.world_size - 1

        self.states = []

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n = 2):
        return random_selection(self.clients, n)

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(Client.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref)

    def clients_ready(self):
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            for client in self.clients:
                if client.name not in ready_clients:
                    responses.append((client, _remote_method_async(Client.is_ready, client.ref)))
            all_ready = True
            for res in responses:
                result = res[1].wait()
                if result:
                    logging.info(f'{res[0]} is ready')
                    ready_clients.append(res[0])
                else:
                    logging.info(f'Waiting for {res[0]}')
                    all_ready = False

            time.sleep(2)
        logging.info('All clients are ready')

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def step(self, client_weights):
        self.states = []
        for i in range(self.device_num):
            self.states.append([])
            for param in client_weights[i].values():
                self.states[i].append(param.flatten())
        self.states = list(zip(*self.states))
        self.attack()
        self.update()
        self.states = []
        return self.model.state_dict()

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
        if trim > 0:
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

    def remote_run_epoch(self, epochs):
        responses = []
        client_weights = []
        selected_clients = self.select_clients(self.config.clients_per_round)
        self.epoch_counter += epochs
        for client in selected_clients:  # de facto sequential execution for CPU acceleration
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs)))
            res = responses[-1]
            epoch_data, weights = res[1].wait()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_weights.append(weights)
        updated_model = self.step(client_weights)

        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=updated_model)))

        for res in responses:
            res[1].wait()
        logging.info('Weights are updated')

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.get_client_datasize, client.ref)))
        for res in responses:
            res[0].data_size = res[1].wait()
            logging.info(f'{res[0]} had a result of datasize={res[0].data_size}')

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.test, client.ref)))

        for res in responses:
            accuracy, loss, class_precision, class_recall = res[1].wait()
            logging.info(f'{res[0]} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}_epochs.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Main loop of the Federator
        :return:
        """
        # # Make sure the clients have loaded all the data
        self.client_load_data()
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        for epoch in range(epoch_to_run):
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()
        logging.info(f'Federator is stopping')

