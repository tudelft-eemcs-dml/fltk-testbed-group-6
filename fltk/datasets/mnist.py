from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler


class MNISTDataset(Dataset):

    def __init__(self, args):
        super(MNISTDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading MNIST train data")


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)
        sampler = DistributedSampler(train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading MNIST test data")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)
        sampler = DistributedSampler(test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading MNIST test data")

        return test_data
