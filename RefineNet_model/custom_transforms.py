import random
import torchvision.transforms.functional as TF

class RandomRotate(object):
    def __init__(self):
        self.angles = [x for x in range(91)]

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        angle = random.choice(self.angles)
        return {'img': TF.rotate(img, angle),
                'mask': TF.rotate(mask, angle)}

class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        return {'img': TF.to_tensor(img),
                'mask': TF.to_tensor(mask)}

class GaussianBlur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        return {'img': TF.gaussian_blur(img=img, kernel_size=self.kernel_size),
                'mask': mask}

class VerticalFlip(object):
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        return {'img': TF.vflip(img),
                'mask': TF.vflip(mask)}

class HorizontalFlip(object):
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        return {'img': TF.hflip(img),
                'mask': TF.hflip(mask)}

class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        return {'img': TF.resize(img=img, size=self.img_size),
                'mask': TF.resize(img=mask, size=self.img_size)}


def get_transformations():
    transform_list = []
    img_size = (371, 462)
    kernel_size = 5
    transform_list.append(Resize(img_size))
    transform_list.append(VerticalFlip())
    transform_list.append(HorizontalFlip())
    transform_list.append(RandomRotate())
    transform_list.append(GaussianBlur(kernel_size))
    transform_list.append(ToTensor())

    return transform_list