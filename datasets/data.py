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
    def __init__(self, root, type, transform=None):

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
        path = self.paths[item].strip()
        img_path = os.path.join(self.root, 'images',path)
        mask_path = os.path.join(self.root, 'masks', path.replace('jpg','png'))
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.paths)

class FreiHANDDataset(Dataset):
    def __init__(self, root, type, transform=None):
        self.root = root
        self.paths = os.listdir(os.path.join(root, 'training', 'mask'))
        size = len(self.paths)
        if type not in ['train', 'val', 'test']:
            raise Exception('Error while initialization. Argument type: {} is invalid. It must be train, val or test'.format(type))
        elif type == 'train':
            self.paths = self.paths[:int(0.8*size)]
        elif type == 'val':
            self.paths = self.paths[int(0.8 * size):int(0.9 * size)]
        else:
            self.paths = self.paths[int(0.9 * size):]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        # id = '0'*(8-len(str(item))) + '{}.jpg'.format(item)
        id = self.paths[item]
        img_path = os.path.join(self.root, 'training', 'rgb', id)
        mask_path = os.path.join(self.root, 'training', 'mask', id)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.paths)


class HOFDataset(Dataset):
    def __init__(self, root, type, transform=None):
        self.root = root
        self.paths = os.listdir(os.path.join(root, 'images_resized'))
        size = len(self.paths)
        if type not in ['train', 'val', 'test']:
            raise Exception('Error while initialization. Argument type: {} is invalid. It must be train, val or test'.format(type))
        elif type == 'train':
            self.paths = self.paths[:int(0.8*size)]
        elif type == 'val':
            self.paths = self.paths[int(0.8 * size):int(0.9 * size)]
        else:
            self.paths = self.paths[int(0.9 * size):]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        path = self.paths[item].replace('.jpg','')
        img_path = os.path.join(self.root, 'images_resized', '{}.jpg'.format(path))
        mask_path = os.path.join(self.root, 'masks', '{}.png'.format(path))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        #ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.paths)

class EGTEAGazePlusDataset(Dataset):
    def __init__(self, root, type, transform=None):
        self.root = root
        self.paths = os.listdir(os.path.join(root, 'Images'))
        size = len(self.paths)
        if type not in ['train', 'val', 'test']:
            raise Exception('Error while initialization. Argument type: {} is invalid. It must be train, val or test'.format(type))
        elif type == 'train':
            self.paths = self.paths[:int(0.8*size)]
        elif type == 'val':
            self.paths = self.paths[int(0.8 * size):int(0.9 * size)]
        else:
            self.paths = self.paths[int(0.9 * size):]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        path = self.paths[item].replace('.jpg', '')
        img_path = os.path.join(self.root, 'Images', '{}.jpg'.format(path))
        mask_path = os.path.join(self.root, 'Masks', '{}.png'.format(path))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}


    def __len__(self):
        return len(self.paths)

class HGR1Dataset(Dataset):
    def __init__(self):
        ...

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...


def test_dataset(dataset):

    item = next(iter(dataset))

    # print(item['img'])
    # print(item['mask'])

    transforms.ToPILImage()(item['img']).show()
    transforms.ToPILImage()(item['mask']).show()

    print(len(dataset))


if __name__=='__main__':
    root = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hand14k'
    datasets = EGTEAGazePlusDataset(root=root, type='test')
    test_dataset(datasets)