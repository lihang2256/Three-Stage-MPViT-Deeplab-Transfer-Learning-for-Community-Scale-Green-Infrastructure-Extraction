import os
import torch.utils.data as data
from PIL import Image
from collections import namedtuple
import numpy as np
from .data.data_deal.utils import label_colormap


class Land2(data.Dataset):
    """Land2 <https://github.com/dronedeploy/dd-ml-segmentation-benchmark> Dataset.

    Parameters:
        - root (string): Root directory of dataset
        - split (string, optional): The image split to use, 'train', 'test' or 'val'
        - transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version
    """
    Land2Class = namedtuple('Land2Class', ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])
    classes = [
          Land2Class('BUILDING',   0, 0, False, (230, 25, 75)),  # 红色
          Land2Class('CLUTTER',    1, 1, False, (145, 30, 180)),  # 暗紫色
          Land2Class('VEGETATION', 2, 2, False, (60, 180, 75)),  # 绿色
          Land2Class('WATER',      3, 3, False, (245, 130, 48)),  # 橙色
          Land2Class('GROUND',     4, 4, False, (255, 255, 255)),  # 白色
          Land2Class('CAR',        5, 5, False, (0, 130, 200)),  # 蓝色
          Land2Class('IGNORE',     6, 255, True, (255, 0, 255)),  # 亮紫色
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append((255, 0, 255))
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)

        self.images_dir = os.path.join(self.root, 'image-chips')
        self.targets_dir = os.path.join(self.root, 'label-chips')
        print(self.images_dir)
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
        # 24bit convert to 8 bit
        # label = np.zeros([target.shape[0], target.shape[1]], dtype=int)
        # for i in range(target.shape[0]):
        #     for j in range(target.shape[1]):
        #         label[i][j] = target[i][j][0]
        # print(cls.id_to_train_id[np.array(label)])
        # return cls.id_to_train_id[np.array(label)]
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 6
        return cls.train_id_to_color[target]

