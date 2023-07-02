import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


from labels import labels


class CityScapesDataset(Dataset):
    def __init__(self, root, transform=None, split='train', label_map='id', crop=True):
        """
        
        """
        self.root = root
        self.transform = transform
        self.label_map = label_map
        self.crop = crop

        self.left_paths = glob(os.path.join(root, 'leftImg8bit', split, '**\*.png'))
        self.mask_paths = glob(os.path.join(root, 'gtFine', split, '**\*gtFine_labelIds.png'))
        self.depth_paths = glob(os.path.join(root, 'crestereo_depth', split, '**\*.npy'))

        sorted(self.left_paths)
        sorted(self.mask_paths)
        sorted(self.depth_paths)

        # get label mappings
        self.id_2_train = {}
        self.id_2_cat = {}
        self.train_2_id = {}
        self.id_2_name = {-1 : 'unlabeled'}
        self.trainid_2_name = {19 : 'unlabeled'} # {255 : 'unlabeled', -1 : 'unlabeled'}

        for lbl in labels:
            self.id_2_train.update({lbl.id : lbl.trainId})
            self.id_2_cat.update({lbl.id : lbl.categoryId})
            if lbl.trainId != 19: # (lbl.trainId > 0) and (lbl.trainId != 255):
                self.trainid_2_name.update({lbl.trainId : lbl.name})
                self.train_2_id.update({lbl.trainId : lbl.id})
            if lbl.id > 0:
                self.id_2_name.update({lbl.id : lbl.name})


    def __getitem__(self, idx):
        left = cv2.cvtColor(cv2.imread(self.left_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.uint8)
        depth = np.load(self.depth_paths[idx]) # data is type float16

        if self.crop:
            left = left[:800, :, :]
            mask = mask[:800, :]
            depth = depth[:800, :]

        # get label id
        if self.label_map == 'id':
            mask[mask==-1] == 0
        elif self.label_map == 'trainId':
            for _id, train_id in self.id_2_train.items():
                mask[mask==_id] = train_id
        elif self.label_map == 'categoryId':
            for _id, train_id in self.id_2_cat.items():
                mask[mask==_id] = train_id

        sample = {'left' : left, 'mask' : mask, 'depth' : depth}

        if self.transform:
            sample = self.transform(sample)

        # ensure that no depth values are less than 0
        depth[depth < 0] = 0

        return sample
    

    def __len__(self):
        return len(self.left_paths)
    


