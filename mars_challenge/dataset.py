import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from raft_utils import InputPadder


class MarsDataset(Dataset):
    def __init__(self, root, transform=None, split='train', splits=[0.8, 0.1, 0.1]):
        self.root = root
        self.transform = transform
        self.split = split.lower()
        self.splits = splits

        self.image_path1 = None
        self.image_path2 = None
        
        if self.split not in ('train', 'valid', 'test'): 
            self.speed_df = None
            self.root = os.path.join(self.root, "test")
        else:
            self.speed_df = pd.read_csv(os.path.join(self.root, "train.txt"), header=None)
            self.root = os.path.join(self.root, "train")

            # split DataFrame manually to avoid data leak
            split_indexes = []
            idx = 0
            for i, s in enumerate(splits):
                s_idx = np.round(len(self.speed_df)*s).astype(int)
                split_indexes.append((idx, idx + s_idx))
                idx += s_idx

                if (self.split == 'train') and (i == 0):
                    self.speed_df = self.speed_df.iloc[split_indexes[i][0]: split_indexes[i][1]]
                    break
                elif (self.split == 'valid') and (i == 1):
                    self.speed_df = self.speed_df.iloc[split_indexes[i][0]: split_indexes[i][1]]
                    break
                elif (self.split == 'test') and (i == 2):
                    self.speed_df = self.speed_df.iloc[split_indexes[i][0]: split_indexes[i][1]]
                    break
        

    def __getitem__(self, idx):
        """ Obtains two seuqential image frames and their correpsonding speed.
            Assumes that the frame filenames are 0 indexed.
            """
        self.image_path1 = os.path.join(self.root, f"frame_{idx}.png")
        self.image_path2 = os.path.join(self.root, f"frame_{idx + 1}.png")

        image1 = cv2.cvtColor(cv2.imread(self.image_path1), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(cv2.imread(self.image_path2), cv2.COLOR_BGR2RGB)

        # no need to transform to a tensor
        # don't 0-1 normalize since model already has normalization
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        if self.split in ('train', 'valid', 'test'):
            speed = torch.as_tensor(self.speed_df.iloc[idx + 1][0]).float()
        else:
            speed = None

        if self.transform:
            # get current random state before first transform
            state = torch.get_rng_state() 
            image1 = self.transform(image1)
            
            # reset random state to that of the previous transform
            torch.set_rng_state(state) 
            image2 = self.transform(image2)


        return (image1, image2), speed


    def __len__(self):
        """ Returns total length minus 1 to account for 
            the fact that two images are needed and we
            don't use the very first speed value.
            """
        return len(self.speed_df) - 1