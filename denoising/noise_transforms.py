'''
This script contains noise transforms for pytorch

'''

import torch


class AddGaussianNoise(object):
    ''' Add Gaussian Noise to Images '''

    def __init__(self, sigma):
        self.sigma = sigma # nosie variance

    def __call__(self, image):
         
        # add Gaussian Noise
        image += torch.normal(0, self.sigma, size=image.shape)
        
        return image


class AddSaltandPepperNoise(object):
    ''' Add Salt and Pepper Noise to Images '''

    def __init__(self, pct):
        ''' pct - pcercent nois coverage ranges from 0 -1 '''
        self.pct_salt = 1 - (pct/2) 
        self.pct_pepper = pct/2

    def __call__(self, image):
         
        sample = torch.rand(image.shape)
        image[sample > self.pct_salt] = 1 # add salt
        image[sample < self.pct_pepper] = 0 # add pepper
        
        return image



class AddSpeckleNoise(object):
    ''' Add Speckle Noise to Images '''

    def __init__(self, sigma):
        # self.c = c
        # self.r = r 
        self.sigma = sigma # noise variance

    def __call__(self, image):
         
        # add speckle noise 

        # this approach has issues with sampling from the Gamma Distribution
        # g = torch.distributions.gamma.Gamma(torch.tensor([self.c]), torch.tensor([self.r]))
        # noise = g.sample((image.shape)).squeeze()

        noise = torch.normal(0, self.sigma, size=image.shape)

        return image + image*noise