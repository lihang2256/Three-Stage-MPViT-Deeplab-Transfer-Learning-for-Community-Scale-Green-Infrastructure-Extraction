import os
import torch.utils.data as data
from PIL import Image
from collections import namedtuple
import numpy as np


class Land(data.Dataset):
    """Land <http://https://rs.sensetime.com//> Dataset.

    Parameters:
        - root (string): Root directory of dataset
        - split (string, optional): The image split to use, 'train', 'test' or 'val'
        - transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version
    """
    LandClass = namedtuple('LandClass', ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])
    classes = [
          LandClass('unlabeled', 0, 255, True, (255, 255, 255)),  # 白
          LandClass('1',         1, 0, False, (128, 0, 0)),  # 暗红
          LandClass('2',         2, 1, False, (0, 255, 0)),  # 亮绿
          LandClass('3',         3, 2, False, (0, 128, 0)),  # 暗绿
          LandClass('4',         4, 3, False, (128, 128, 128)),  # 灰
          LandClass('5',         5, 4, False, (0, 0, 255)),  # 蓝
          LandClass('6',         6, 5, False, (255, 0, 0)),  # 亮红
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append((255, 255, 255))
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)

        self.images_dir = os.path.join(self.root, split)
        self.targets_dir = os.path.join(self.root, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []
        if split not in ['train', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train" or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete.')

        img_dir = os.path.join(self.images_dir, 'im1')
        # 只能是伪彩图
        target_dir = os.path.join(self.targets_dir, 'label1')

        for file_name in os.listdir(img_dir):
            self.images.append(os.path.join(img_dir, file_name))
            self.targets.append(os.path.join(target_dir, file_name))

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
        target[target == 255] = 6
        return cls.train_id_to_color[target]


