import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class NyuDataset(Dataset):
    def __init__(self, root, topk_labels=None, transform=None, normalize=False, depth_norm=10):
        """
            NYU Depth Dataset 
            root - path to NYU dataset .mat file
            topk_labels - (list) top k classes for segmentation
            transform - transforms for segmentation labels and depth maps
            normalize - determines whether to apply ImageNet normalization to RGB images
            depth_norm - Normalization for depth map based on training data
            extra_augs - determines whether to apply extra agumentations to RGB images
        """
        self.topk_labels = topk_labels
        self.transform = transform

        # image net normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std) if normalize else None

        # arbitrary normalization factor for depth based on train data
        self.depth_norm = depth_norm

        # open .mat file as an h5 object
        self.h5_obj = h5py.File(root, mode='r')

        # obtain desired groups
        self.images = self.h5_obj['images'] # rgb images
        self.depths = self.h5_obj['depths'] # depths
        self.labels = self.h5_obj['labels'] # sematic class mask for each image
        self.names = self.h5_obj['names']   # sematic class labels
        # self.instances = self.h5_obj['instances'] # instances
        # self.namesToIds = self.h5_obj['namesToIds']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.rotate(self.images[idx].transpose(1, 2, 0), cv2.ROTATE_90_CLOCKWISE) # rgb image
        depth = cv2.rotate(self.depths[idx], cv2.ROTATE_90_CLOCKWISE) # depth map
        label = cv2.rotate(self.labels[idx], cv2.ROTATE_90_CLOCKWISE).astype(np.float32) # semantic segmentation label

        # reduce to topk labels (by placing them in the uncategorized class)
        if self.topk_labels:
            for lbl in np.unique(label).astype(int):
                if lbl not in self.topk_labels:
                    label[label == lbl] = 0

        if self.transform:
            """ We need to apply the same random transforms to the image and mask,
                typically we would just place everything in a dict and use custom
                transform classes. However, we have a limited amount of training 
                data and will likely want to add aggressive augmentation. Instead
                we will get current random state before first transform and update 
                the state before each subsequent transform.
            """
            state = torch.get_rng_state() 
            image = self.transform(image)

            # reset random state to that of the previous transform
            torch.set_rng_state(state) 
            depth = self.transform(depth)

            # reset random state to that of the previous transform
            torch.set_rng_state(state) 
            label = self.transform(label)

        # apply normalizations
        if self.normalize:
            image = self.normalize(image)
            depth = depth/self.depth_norm
        
        return image, (depth, label)
    
    
    def str_label(self, idx):
        """ 
            Obtains string label for a class index. Names/Labels are indexed from 1,
            this function is able to take this into account by subtracting 1.
            In the NYU depth dataset, labels equal 0 are considered unlabeled.
        """
        if idx - 1 < 0:
            return 'unlabeled'
        return ''.join(chr(i[0]) for i in self.h5_obj[self.names[0, idx - 1]])

    def close(self):
        self.h5_obj.close()

    def __exit__(self, *args):
        self.close()
    