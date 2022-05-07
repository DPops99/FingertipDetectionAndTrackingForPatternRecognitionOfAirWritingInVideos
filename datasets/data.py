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
    def __init__(self, root, type, transform):
        self.root = root
        if type not in ['train', 'val']:
            raise Exception('Error while initialization. Argument type: {} is invalid. It must be train or val'.format(type))
        elif type=='train':
            self.type = 'traning'
        else:
            #ADD EVALUATION DATASET WITH ANNOTATIONS IN THE EVALUATION DIRECTORY !!!!!!!!!!!!!!!!!!!!!!!!!
            self.type = 'evaluation'
        self.len = len(os.listdir(os.path.join(root, type, 'mask')))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        id = '0'*(8-len(str(item))) + '{}.jpg'.format(item)
        img_path = os.path.join(self.root, self.type, id)
        mask_path = os.path.join(self.root, self.type, id)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        ...


class HOFDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.len = len(os.listdir(os.path.join(root, 'images_resized')))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        img_path = os.path.join(self.root, 'images_resized', '{}.jpg'.format(item+1))
        mask_path = os.path.join(self.root, 'masks', '{}.jpg'.format(item + 1))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return self.len

class EGTEAGazePlusDataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...


if __name__=='__main__':
    item = 16
    id = '0'*(8-len(str(item))) + '{}.jpg'.format(item)
    print(id)