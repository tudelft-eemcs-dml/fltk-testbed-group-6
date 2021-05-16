from fltk.datasets.distributed import DistDataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler

class DistMNISTDataset(DistDataset):

    def __init__(self, args):
        super(DistMNISTDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' MNIST train data")

        self.train_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor()]))
        self.train_sampler = DistributedSampler(self.train_dataset, rank=self.args.get_rank(),
                                                num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' MNIST test data")
        self.test_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=False, download=True,
                                           transform=transforms.Compose([transforms.ToTensor()]))
        self.test_sampler = DistributedSampler(self.test_dataset, rank=self.args.get_rank(),
                                               num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading MNIST train data")

        train_dataset = datasets.MNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading MNIST test data")

        test_dataset = datasets.MNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading MNIST test data")

        return test_data
