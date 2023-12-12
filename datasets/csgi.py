import os
import torch.utils.data as data
from PIL import Image
from collections import namedtuple
import numpy as np


class CSGI(data.Dataset):
    """CEF Dataset.

    Parameters:
        - root (string): Root directory of dataset
        - split (string, optional): The image split to use, 'train', 'test' or 'val'
        - transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version
    """
    CSGIClass = namedtuple('CEFClass', ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])
    classes = [
        CSGIClass('_background_', 0, 255, False, (0, 0, 0)),    # 黑色
        CSGIClass('Bush area', 1, 0, False, (128, 0, 0)),    # 暗红
        CSGIClass('Grass', 2, 1, False, (0, 128, 0)),    # 暗绿
        CSGIClass('Lake', 3, 2, False, (128, 128, 0)),  # 暗黄 / name:Pool
        CSGIClass('Terrace greenery', 4, 3, False, (0, 0, 128)),    # 暗蓝
        CSGIClass('Tree', 5, 4, False, (128, 0, 128)),  # 紫色
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    # train_id_to_color = [c.color for c in classes if (c.train_id != -1)]
    train_id_to_color.append((0, 0, 0))
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    # print(id_to_train_id)

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)

        self.images_dir = os.path.join(self.root, 'image-chips')
        self.targets_dir = os.path.join(self.root, 'label-chips')
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split in ['train', 'val', 'test']:
            split_f = os.path.join(self.root, split + '.txt')
            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                self.images = [os.path.join(self.images_dir, x) for x in file_names]
                self.targets = [os.path.join(self.targets_dir, x) for x in file_names]
        else:
            raise ValueError(
                'Wrong image_set entered! Please use split="train" or split="val"')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        target = self.encode_target(target)
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 5
        return cls.train_id_to_color[target]
