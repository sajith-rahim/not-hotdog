import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class ImageNetDataset(Dataset):
    def __init__(self, paths, augmentations=None):
        self.paths = paths
        self.augmentations = augmentations

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 0 if 'nothotdog' in str(path) else 1

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        else:
            transforms_ = transforms.Compose([
                transforms.RandomResizedCrop(300),
                transforms.RandomRotation(30),
                transforms.ColorJitter(),
                transforms.ToTensor()
            ])
            image = Image.fromarray(image.astype('float32'), 'RGB')
            # image = torch.tensor(image, dtype=torch.float32)
            augmented = transforms_(img=image)
            image = augmented

        #print(image.shape)
        return image, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def load_img(path, augmentations=None):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300, 300))
        if augmentations:
            augmented = augmentations(image=image)
            image = augmented['image']
        else:
            transforms_ = transforms.Compose([
                transforms.ToTensor()
            ])
            image = Image.fromarray(image.astype('float32'), 'RGB')
            augmented = transforms_(img=image)
            image = augmented.unsqueeze(0)
        return image
