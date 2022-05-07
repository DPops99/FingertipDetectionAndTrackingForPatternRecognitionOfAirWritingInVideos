import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms

class EgoHandsDataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...

class EgoYouTubeHandsDataset(Dataset):
    def __init__(self, root, type, transform):

        if type not in ['train', 'val','test']:
            raise Exception('Error while initialization. Argument type: {} is invalid. It must be train, val or test'.format(type))

        self.root = root
        with open(os.path.join(root, 'train-val-test-split','{}.txt'.format(type))) as f:
            self.paths = f.readlines()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform


    def __getitem__(self, item):
        path = self.paths[item]
        img_path = os.path.join(self.root, 'images',path)
        mask_path = os.path.join(self.root, 'masks', path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.paths)

class FreiHANDDataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...


class HOFDataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...

class EGTEAGazePlusDataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...