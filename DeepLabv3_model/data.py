import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, filepath, type='train'):
        self.filepath = filepath
        imgs_path = os.path.join(self.filepath, 'hgr1_images', 'original_images')

        self.imgs = []

        for _,_,files in sorted(os.walk(imgs_path)):
            for file in files:
                self.imgs.append(file)

        if type == 'train':
            self.imgs = self.imgs[:int(0.8 * len(self.imgs))]
        elif type == 'valid':
            self.imgs = self.imgs[int(0.8 * len(self.imgs)):int(0.9 * len(self.imgs))]
        elif type == 'test':
            self.imgs = self.imgs[int(0.9 * len(self.imgs)):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        width = 320
        height = 480
        dim = ( height, width)
        img_path = os.path.join(self.filepath, 'hgr1_images', 'original_images',self.imgs[item])
        mask_path = os.path.join(self.filepath, 'hgr1_skin', 'skin_masks',self.imgs[item].replace('.jpg','.bmp'))
        mask_path = mask_path.replace('.JPG', '.bmp')
        # img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path)
        # print(f'img shape before convert: {img.shape}')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(f'img shape after convert: {img.shape}')
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #
        # return transform(img, mask)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        preprocess_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=dim),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        preprocess_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=dim)
        ])

        img = preprocess_img(img)
        mask = preprocess_mask(mask)

        input = {
            'img': img,
            'mask': mask
        }

        return input



if __name__=="__main__":
    filepath = "/home/popa/Documents/diplomski_rad/FingertipDetectionAndTrackingForPatternRecognitionOfAirWritingInVideos/segmentation_dataset/hgr1"
    train_dataset = SegmentationDataset(filepath, type='train')
    test_dataset = SegmentationDataset(filepath, type='test')
    valid_dataset = SegmentationDataset(filepath, type='valid')

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(valid_dataset))




