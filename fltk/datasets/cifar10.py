from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler

from fltk.util.data_sampler_utils import LimitLabelsSampler


class CIFAR10Dataset(Dataset):

    def __init__(self, args):
        super(CIFAR10Dataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)
        sampler = self.get_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 train data")

        return train_data        

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)
        sampler = self.get_sampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")

        return test_data

    def get_sampler(self, dataset):
        sampler = None
        if self.args.get_distributed():
            method = self.args.get_sampler()
            if method == "uniform":
                sampler = DistributedSampler(dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            elif method == "limit labels":
                sampler = LimitLabelsSampler(dataset, self.args.get_rank(), self.args.get_world_size, *self.args.get_sampler_args())
            else:   # default
                self.get_args().get_logger().warning("Unknown sampler " + method + ", using uniform instead")
                sampler = DistributedSampler(dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())    

        return sampler