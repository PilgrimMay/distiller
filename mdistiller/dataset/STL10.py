import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class STL10Instance(datasets.STL10):
    """STL10Instance Dataset"""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_STL10_train_transform():
    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize([32,32]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
        ]
    )

    return train_transform

def get_STL10_test_transform():
    return transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

# 0.4467106 0.43980986 0.40664646
# 0.22414584 0.22148906 0.22389975


def get_STL10_dataloaders(batch_size, val_batch_size, num_workers):
    print("batch size:",batch_size)
    data_folder = get_data_folder()
    train_transform = get_STL10_train_transform()
    test_transform = get_STL10_test_transform()
    train_set = STL10Instance(
        root=data_folder, download=True, split='train', transform=train_transform
    )
    # num_data = len(train_set)
    test_set = datasets.STL10(
        root=data_folder, download=True, split='test', transform=test_transform
    )
    num_data = len(test_set)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data





