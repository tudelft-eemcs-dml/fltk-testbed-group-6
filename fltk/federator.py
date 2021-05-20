import os
import datetime
import time
from typing import List

import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
from math import sqrt

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

    def __init__(self, client_id_triple, num_epochs=3, config=None):
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
        self.rule = config.aggregation_rule
        self.attack_type = config.attack_type
        self.compromised = config.compromised_num
        if self.rule == 'krum':
            self.lambda_threshold = 1e-5
            self.first_compromised_model = None
            self.begin_searching_lambda = False
        self.device_num = self.config.world_size - 1
        logging.info(f'Attacking rule: {self.rule}')
        self.states = []
        self._epoch = 0

        self.loss_function = self.args.get_loss_function()()
        self.device = None
        self.init_device()
        self.dataset = None
        self.init_dataloader()

    def init_dataloader(self):
        self.args.distributed = True
        self.args.rank = 0
        self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)

    def init_device(self):
        if self.args.cuda and torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n=2):
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
        if self.rule == 'trimmed' or self.rule == 'median':
            for i in range(self.device_num):
                self.states.append([])
                for param in client_weights[i].values():
                    self.states[i].append(param.flatten())
            self.states = list(zip(*self.states))
            for i in range(len(self.states)):
                self.states[i] = torch.stack(self.states[i])
        elif self.rule == 'krum':
            self.states = client_weights
        logging.info(f"round {self._epoch} state saved")
        self.attack()
        logging.info(f"round {self._epoch} attack finished")
        self.update()
        logging.info(f"round {self._epoch} update finished")
        self.states = []
        return self.model.state_dict()

    def attack(self):
        if self.attack_type == 'flip':
            return
        elif self.attack_type == 'no':
            if self.rule == 'krum':
                self.flatten_states = []
                for params in self.states:
                    weight = params['linear.weight'].flatten()
                    bias = params['linear.bias']
                    flatten_params = torch.cat((weight, bias))
                    self.flatten_states.append(flatten_params)

                weight = self.model.state_dict()['linear.weight'].flatten()
                bias = self.model.state_dict()['linear.bias']
                self.flatten_global = torch.cat((weight, bias))
            return

        if self.rule == 'trimmed' or self.rule == 'median':
            i = 0
            for v in self.model.state_dict().values():
                v = v.flatten()
                for j in range(len(self.states[i][0])):
                    result = self._attack(self.states[i][:, j], v[j])
                    for k in range(len(result)):
                        self.states[i][k][j] = result[k]
                i += 1
        elif self.rule == 'krum':
            # for now only works in minst_lr
            self.flatten_states = []
            for params in self.states:
                weight = params['linear.weight'].flatten()
                bias = params['linear.bias']
                flatten_params = torch.cat((weight, bias))
                self.flatten_states.append(flatten_params)

            weight = self.model.state_dict()['linear.weight'].flatten()
            bias = self.model.state_dict()['linear.bias']
            self.flatten_global = torch.cat((weight, bias))

            logging.info(f'flatten global: {self.flatten_global}')

            self.compromised_states = []
            self.first_compromised_model = self._attack(None, None)
            self.compromised_states.append(self.first_compromised_model)
            for i in range(1, len(self.flatten_states)):
                if i <= self.compromised - 1:
                    result = self.first_compromised_model  # would be changed later
                else:
                    result = self.flatten_states[i]
                self.compromised_states.append(result)

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

    def krum(self, compromised_states):
        """
        select the local model with the smallest sum of distance as global model
        :param compromised_states: states containing compromised states (variable length)
        :return: selected model
        """

        n = self.device_num - self.compromised - 2
        if n < 0:
            logging.warning('Too many compromised workers, c should be < (m-2)/2')
            n = 0

        records = {}
        for i in range(len(compromised_states)):
            state = compromised_states[i]
            dists = []
            for another_state in compromised_states:
                if torch.all(state.eq(another_state)): continue
                euclidean_dist = torch.sum(((state - another_state) ** 2))
                dists.append(euclidean_dist)
            sorted_dists = sorted(dists)
            records[i] = sum(sorted_dists[:n])
        self.sorted_records = dict(sorted(records.items(), key=lambda item: (item[1].item(), item[0])))  # also used for calculating epsilon
        index = next(iter(self.sorted_records))
        logging.info('Model selected by Krum: ' + str(index+1))

        return index+1, compromised_states[index]

    def krum_changing_direction(self, states):
        _, global_model = self.krum(states)
        previous_global = self.flatten_global
        s = torch.where(global_model > previous_global, 1, -1)
        return s

    def krum_lambda(self):
        '''
        calculate the initial value of lambda, only do this at the beginning of each iteration
        :return:
        '''

        d = self.flatten_global.shape[0]
        lambda_ = 1/((self.device_num - 2*self.compromised - 1)*sqrt(d))

        n = self.device_num - self.compromised - 2
        if n < 0:
            logging.warning('Too many compromised workers, c should be < (m-2)/2')
            n = 1

        if self.attack_type == 'full':
            benign_dists = dict()
            for i in range(self.compromised, len(self.flatten_states)):
                benign_dists[i] = []
                for j in range(self.compromised, len(self.flatten_states)):
                    if i == j:
                        continue
                    else:
                        euclidean_dist = torch.sum(((self.flatten_states[i] - self.flatten_states[j]) ** 2))
                        benign_dists[i].append(euclidean_dist)
            dists_sum_list = []
            for k, l in benign_dists.items():
                sorted_l = sorted(l)
                dists_sum_list.append(sum(sorted_l[:n]))
            smallest_benign_dist = sorted(dists_sum_list)[0]
            lambda_ *= smallest_benign_dist

            largest_benign_dist = 0.0
            for i in range(self.compromised, len(self.flatten_states)):
                for j in range(self.compromised, len(self.flatten_states)):
                    if i == j:
                        continue
                    else:
                        euclidean_dist = torch.sum(((self.flatten_states[i] - self.flatten_states[j]) ** 2))
                        if euclidean_dist > largest_benign_dist:
                            largest_benign_dist = euclidean_dist
            lambda_ += 1 / sqrt(d) * largest_benign_dist
        elif self.attack_type == 'partial':
            benign_dists = dict()
            for i in range(0, self.compromised):  # use before-attack compromised models as benign models
                benign_dists[i] = []
                for j in range(0, self.compromised):
                    if i == j:
                        continue
                    else:
                        euclidean_dist = torch.sum(((self.flatten_states[i] - self.flatten_states[j]) ** 2))
                        benign_dists[i].append(euclidean_dist)
            dists_sum_list = []
            for k, l in benign_dists.items():
                sorted_l = sorted(l)
                dists_sum_list.append(sum(sorted_l[:n]))
            smallest_benign_dist = sorted(dists_sum_list)[0]
            lambda_ *= smallest_benign_dist

            largest_benign_dist = 0.0
            for i in range(0, self.compromised):
                for j in range(0, self.compromised):
                    if i == j:
                        continue
                    else:
                        euclidean_dist = torch.sum(((self.flatten_states[i] - self.flatten_states[j]) ** 2))
                        if euclidean_dist > largest_benign_dist:
                            largest_benign_dist = euclidean_dist
            lambda_ += 1 / sqrt(d) * largest_benign_dist

        logging.info(f'Initial lambda: {lambda_}')

        return lambda_

    def krum_reverse_flatten_lr(self, params, input_dim=784, n_classes=10):
        bias = params[-n_classes:]
        weights = params[:-n_classes].view(n_classes, input_dim)
        return weights, bias

    def poisoning(self, params, original, full):
        if self.rule == 'trimmed' or self.rule == 'median':
            params = torch.Tensor(params)
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
        elif self.rule == 'krum':
            if full:
                # 1. w_1' = w_Re - lambda * s
                # 2. the other c-1 to be more close to w_1', distance at most epsilon (incomplete)
                s = self.krum_changing_direction(self.flatten_states)
                w_Re = self.flatten_global
                if not self.begin_searching_lambda:
                    self.lambda_ = self.krum_lambda()
                    self.begin_searching_lambda = True
                compromised_w_1 = w_Re - self.lambda_ * s
                return compromised_w_1
            else:
                s = self.krum_changing_direction(self.flatten_states[:self.compromised])
                w_Re = self.flatten_global
                if not self.begin_searching_lambda:
                    self.lambda_ = self.krum_lambda()
                    self.begin_searching_lambda = True
                compromised_w_1 = w_Re - self.lambda_ * s
                return compromised_w_1

    def aggregate(self, params):
        if self.rule == 'trimmed':
            return self.trimmed(params, self.compromised)
        if self.rule == 'median':
            return self.trimmed(params, (len(params) - 1) // 2)
        return params[0]

    def update(self):
        if self.rule == 'trimmed' or self.rule == 'median':
            d = {}
            i = 0
            for k, v in self.model.state_dict().items():
                t = torch.zeros_like(self.states[i][0])
                for j in range(t.shape[0]):
                    t[j] = self.aggregate(self.states[i][:, j])
                d[k] = t.reshape_as(v)
                d[k].requires_grad = v.requires_grad
                i += 1
        elif self.rule == 'krum':
            d = {}
            if self.attack_type == 'full':
                index, flatten_params = self.krum(self.compromised_states)
                while index != 1:
                    self.lambda_ *= 1/2
                    if self.lambda_ < self.lambda_threshold:
                        break
                    else:
                        new_compromised_w_1 = self._attack(None, None)
                        current_compromised_states = []
                        for i in range(self.compromised):
                            current_compromised_states.append(new_compromised_w_1)
                        for i in range(self.compromised, self.device_num):
                            current_compromised_states.append(self.compromised_states[i])
                        index, flatten_params = self.krum(current_compromised_states)
                logging.info(f'lambda: {self.lambda_}')
                new_compromised_w_1 = self._attack(None, None)  # calculated with the solved lambda
            elif self.attack_type == 'partial':
                current_compromised_states = [self.first_compromised_model]
                for i in range(1, self.compromised):
                    current_compromised_states.append(self.flatten_states[i])  # use before attack compromised models as benign models
                index, flatten_params = self.krum(current_compromised_states)
                num_compromised_models = 1
                while index != 1:
                    self.lambda_ *= 1 / 2
                    if self.lambda_ < self.lambda_threshold:
                        num_compromised_models += 1

                    new_compromised_w_1 = self._attack(None, None)
                    current_compromised_states = []
                    for _ in range(num_compromised_models):
                        current_compromised_states.append(new_compromised_w_1)
                    for i in range(self.compromised):
                        current_compromised_states.append(self.flatten_states[i])
                    index, flatten_params = self.krum(current_compromised_states)

                logging.info(f'lambda: {self.lambda_}')
                new_compromised_w_1 = self._attack(None, None)  # calculated with the solved lambda
            elif self.attack_type == 'no':
                index, flatten_params = self.krum(self.flatten_states)
                new_compromised_w_1 = flatten_params

            d['linear.weight'], d['linear.bias'] = self.krum_reverse_flatten_lr(new_compromised_w_1)

        self.model.load_state_dict(d)
        logging.info(f'update: {self.model.state_dict()}')

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

    def test_global_model(self):
        """
        Test global model

        :return:
        """
        # self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)
        logging.info(f'rank: {self.args.get_rank()}')  # group 0 should have mixed classes
        self.model.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        logging.info(self.model.state_dict())

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

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
            logging.info(f'{res[0]} had a train loss of {epoch_data.loss_train}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_weights.append(weights)
        updated_model = self.step(client_weights)
        self.update_local(updated_model)

    def update_local(self, updated_model):

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
        self.update_local(self.model.state_dict())
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        for epoch in range(epoch_to_run):
            self._epoch = epoch + 1
            if epoch % 10 == 0:
                self.test_global_model()
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()

        results = self.test_global_model()
        logging.info(repr(results))
        logging.info(f'Federator is stopping')

