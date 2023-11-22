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
        ''' pct - pcercent noise coverage ranges from 0 -1 '''
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
    

class MotionBlur():
    ''' Induces Motion Blurr on a given image
        Float64 and Complex128 are required dtypes
        Inputs:
            a - horizontal motion factor range: (1e-6 - 0.2)
            b - vertical motion factor range: (1e-6 - 0.2)
        '''

    def __init__(self, a, b):
        self.a = abs(a)
        self.b = abs(b)

    def __call__(self, image):
        if (self.a == 0) and (self.b == 0):
            return image 
        
        btch, c, n, m = image.shape
        
        # get values for a and b
        self.a = torch.distributions.Uniform(-self.a, self.a).sample((btch, 1)).double() + 1e-6
        self.b = torch.distributions.Uniform(-self.b, self.b).sample((btch, 1)).double() + 1e-6
    
        # compute FFT of image
        image = image.double()
        F = torch.fft.fftshift(torch.fft.fft2(image), dim=(-1,-2))

        # compute motion blurr function H in Frequency Domain
        u_index, v_index = torch.meshgrid(torch.arange(-(n//2), (n//2)), torch.arange(-(m//2), (m//2)))
        u_index = u_index.reshape((-1, 1)).repeat(1, btch).T.double()
        v_index = v_index.reshape((-1, 1)).repeat(1, btch).T.double()
       
        omega = torch.pi*(u_index*self.a + v_index*self.b)
        H = (1/omega) * torch.sin(omega) * torch.exp(-(1j * omega))

        # remove NaN
        H[torch.isnan(H)] = 1.0 + 1.0j

        H /= torch.abs(H).max()
        H = H.reshape((btch, n, m))

        # perform motion blurring in Frequency Domain for each channel
        G = torch.zeros_like(F)
        for i in range(c):
            G[:, i, :, :] = F[:, i, :, :]*H

        # get blurred image 
        return torch.abs(torch.fft.ifft2(G))