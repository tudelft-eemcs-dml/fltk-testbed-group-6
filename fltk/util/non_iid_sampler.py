from torch.utils.data import Dataset
import random
import logging
from torch.utils.data import DistributedSampler
from typing import Iterator


class PDegreeSampler(DistributedSampler):
    """
        p degree sampler as stated in local poisoning paper
    """

    def __init__(self, dataset, rank, world_size, group, p=0.5, seed=42):
        super(PDegreeSampler, self).__init__(
            dataset, world_size, rank, False)

        client_id = rank - 1
        n_clients = world_size - 1
        n_groups = n_clients
        # order the indices by label
        ordered_by_label = [[] for i in range(len(dataset.classes))]
        for index, target in enumerate(dataset.targets):
            ordered_by_label[target].append(index)

        n_labels = len(ordered_by_label)

        labels = list(range(n_labels))  # list of labels to distribute
        clients = list(range(n_clients))  # keeps track of which clients should still be given a label
        random.seed(seed)  # seed, such that the same result can be obtained multiple times

        indices = []
        for label in labels:
            interval = int(len(ordered_by_label[label]) / n_clients)
            if rank == 0:  # federator has all data
                start_index = 0
                end_index = len(ordered_by_label[label])
                indices += ordered_by_label[label][start_index:end_index]
            else:
                index = clients.index(client_id)  # find the position of this client
                if label % n_groups == group - 1:
                    random_list = [0] * int((1-p)*interval) + [1] * (interval-int((1-p)*interval))
                    random.shuffle(random_list)
                    indices += [data for data, mask in zip(ordered_by_label[label][interval*index:interval*(index+1)], random_list) if mask]
                else:
                    indices += ordered_by_label[label][interval*index:interval*(index+1)]

        random.seed(seed + client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)