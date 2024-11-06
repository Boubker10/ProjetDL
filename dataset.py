import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List


class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug=None) -> None:
        self.dataset = []
        self.root_dir = root_dir
        self.__add_dataset__("bean", [1, 0, 0, 0, 0, 0])
        self.__add_dataset__("carrot", [0, 1, 0, 0, 0, 0])
        self.__add_dataset__("cucumber", [0, 0, 1, 0, 0, 0])
        self.__add_dataset__("potato", [0, 0, 0, 1, 0, 0])
        self.__add_dataset__("tomato", [0, 0, 0, 0, 1, 0])
        self.__add_dataset__("broccoli", [0, 0, 0, 0, 0, 1])
        if aug is None:
            self.augmentation = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor() 
            ])
        else:
            self.augmentation = aug
    
    def __add_dataset__(self, dir_name: str, class_label: List[int]) -> None:
            full_path = os.path.join(self.root_dir, dir_name)
            label = np.array(class_label)
            for fname in os.listdir(full_path):
                fpath = os.path.join(full_path, fname)
                fpath = os.path.abspath(fpath)
                self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]
        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)
        label = torch.Tensor(label)

        return image, label
