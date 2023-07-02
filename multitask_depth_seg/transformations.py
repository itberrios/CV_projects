# tranforms
# main idea for transforms, create classes for them so we can use pytorch with them
# use each transform class to wrap what is already implemented in either pytorch or albumentations

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
# from albumentations.augmentations.transforms import RandomShadow

class Normalize(object):
    """ Normalizes RGB image to  0-mean 1-std_dev """ 
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], depth_norm=5, max_depth=250):
        self.mean = mean
        self.std = std
        self.depth_norm = depth_norm
        self.max_depth = max_depth

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']
            
        return {'left': TF.normalize(left, self.mean, self.std), 
                'mask': mask, 
                'depth' : torch.log(torch.clip(depth, 0, self.max_depth))/self.depth_norm}


class AddColorJitter(object):
    """Convert a color image to grayscale and normalize the color range to [0,1].""" 
    def __init__(self, brightness, contrast, saturation, hue):
        ''' Applies brightness, constrast, saturation, and hue jitter to image ''' 
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        return {'left': self.color_jitter(left), 
                'mask': mask, 
                'depth' : depth}


class Rescale(object):
    """ Rescales images with bilinear interpolation and masks with nearest interpolation """

    def __init__(self, h, w):
        self.h, self.w = h, w

        # state = torch.get_rng_state() 
        # self.rescale = transforms.Resize((self.h, self.w))

        # # get  same transform for mask
        # torch.set_rng_state(state)
        # self.mask_rescale = transforms.Resize((self.h, self.w), transforms.InterpolationMode.NEAREST)

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        return {'left': TF.resize(left, (self.h, self.w)), 
                'mask': TF.resize(mask.unsqueeze(0), (self.h, self.w), transforms.InterpolationMode.NEAREST), 
                'depth' : TF.resize(depth.unsqueeze(0), (self.h, self.w))}


class RandomCrop(object):
    def __init__(self, h, w, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.h = h
        self.w = w
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']
        i, j, h, w = transforms.RandomResizedCrop.get_params(left, scale=self.scale, ratio=self.ratio)

        return {'left': TF.resized_crop(left, i, j, h, w, (self.h, self.w)), 
                'mask': TF.resized_crop(mask.unsqueeze(0), i, j, h, w, (self.h, self.w), interpolation=TF.InterpolationMode.NEAREST),
                'depth' : TF.resized_crop(depth.unsqueeze(0), i, j, h, w, (self.h, self.w))}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
         
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        return {'left': transforms.ToTensor()(left), 
                'mask': torch.as_tensor(mask, dtype=torch.int64),
                'depth' : transforms.ToTensor()(depth).type(torch.float32)}
    

# class ElasticTransform(object):
#     def __init__(self, alpha=25.0, sigma=5.0):
#         self.alpha = [1.0, alpha]
#         self.sigma = [1, sigma]

#     def __call__(self, sample):
#         left, mask, depth = sample['left'], sample['mask'], sample['depth']
#         H, W = mask.shape

#         displacement = transforms.ElasticTransform.get_params(self.alpha, self.sigma, [H, W])

#         return {'left': TF.elastic_transform(left, displacement), 
#                 'mask': TF.elastic_transform(mask.unsqueeze(0), displacement, interpolation=TF.InterpolationMode.NEAREST), 
#                 'depth' : TF.elastic_transform(depth, displacement)}
    

# new transform to rotate the images
class RandomRotate(object):
    def __init__(self, angle):
        if not isinstance(angle, (list, tuple)):
            self.angle = (-abs(angle), abs(angle))
        else:
            self.angle = angle

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        angle = transforms.RandomRotation.get_params(self.angle)

        return {'left': TF.rotate(left, angle), 
                'mask': TF.rotate(mask.unsqueeze(0), angle), 
                'depth' : TF.rotate(depth, angle)}
    
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        if torch.rand(1) < self.prob:
            return {'left': TF.hflip(left), 
                    'mask': TF.hflip(mask), 
                    'depth' : TF.hflip(depth)}
        
        else:
            return {'left': left, 
                    'mask': mask, 
                    'depth' : depth}
        

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        left, mask, depth = sample['left'], sample['mask'], sample['depth']

        if torch.rand(1) < self.prob:
            return {'left': TF.vflip(left), 
                    'mask': TF.vflip(mask), 
                    'depth' : TF.vflip(depth)}
        
        else:
            return {'left': left, 
                    'mask': mask, 
                    'depth' : depth}