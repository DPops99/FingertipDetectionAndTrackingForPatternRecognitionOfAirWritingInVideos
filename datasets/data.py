import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
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
        self.img_size = (371, 462)
        with open(os.path.join(root, 'train-val-test-split','{}.txt'.format(type))) as f:
            self.paths = f.readlines()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
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
        self.img_size = (371, 462)
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
                transforms.Resize(self.img_size),
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
        self.img_size = (371, 462)
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
                transforms.Resize(self.img_size),
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
        self.img_size = (371, 462)
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
                transforms.Resize(self.img_size),
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
    def __init__(self, root, type, transform=None):
        self.root = root
        self.paths = os.listdir(os.path.join(root, 'hgr1_images', 'original_images'))
        size = len(self.paths)
        self.img_size = (371,462)
        if type not in ['train', 'val', 'test']:
            raise Exception(
                'Error while initialization. Argument type: {} is invalid. It must be train, val or test'.format(type))
        elif type == 'train':
            self.paths = self.paths[:int(0.8 * size)]
        elif type == 'val':
            self.paths = self.paths[int(0.8 * size):int(0.9 * size)]
        else:
            self.paths = self.paths[int(0.9 * size):]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        path = self.paths[item].replace('.jpg', '')
        img_path = os.path.join(self.root, 'hgr1_images', 'original_images', '{}.jpg'.format(path))
        mask_path = os.path.join(self.root, 'hgr1_skin', 'skin_masks', '{}.bmp'.format(path))

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # ADD TRANSFORMATIONS !!!!!!!!!

        img = self.transform(img)
        mask = self.transform(mask)

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.paths)


def test_dataset():
    hgr1_root = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hgr1'
    hof_root = '/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hand_over_face_corrected/hand_over_face'
    list_datasets = [HGR1Dataset(root=hgr1_root, type='train'), HOFDataset(root=hof_root, type='train')]
    final_dataset = ConcatDataset(list_datasets)
    final_dataloader = DataLoader(final_dataset, batch_size=2, shuffle=True)

    i = 0

    for batch in final_dataloader:
        if i > 2:
            break
        imgs = batch['img']
        masks = batch['mask']
        for mask in masks:
            transforms.ToPILImage()(mask).show()
        i += 1





if __name__=='__main__':
    test_dataset()