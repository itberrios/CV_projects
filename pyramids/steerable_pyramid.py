"""
 Regular, Complex, and Sub-octave Complex Steerable Pyramids

 Sources:
    Papers: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
        - https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
        - http://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf
    Code: 
        - http://people.csail.mit.edu/nwadhwa/phase-video/
        - https://github.com/LabForComputationalVision/matlabPyrTools
        - https://github.com/LabForComputationalVision/pyrtools
    Misc:
        - https://rafat.github.io/sites/wavebook/advanced/steer.html
        - http://www.cns.nyu.edu/~eero/steerpyr/
        - https://www.cns.nyu.edu/pub/lcv/simoncelli90.pdf
        - http://www.cns.nyu.edu/~eero/imrep-course/Slides/07-multiScale.pdf
"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from pyramid_utils import *


## Steerable Pyramid Base class
class SteerablePyramid():
    def __init__(self, depth, nbands, twidth=1, complex_pyr=False):
        """ 
            depth - Pyramid Depth (height)
            nbands - number of orientations
            twidth - transition width (controls abruptness of the falloff)  
            complex_pyr - determines whether to create a Complex Pyramid
        """
        # max_depth = int(np.floor(np.log2(np.min(np.array(image.shape)))) - 2)
        self.depth = depth
        self.nbands = nbands # number of orientations
        self.twidth = twidth # width of lo/hi transition region
        self.complex_pyr = complex_pyr


    def _get_radial_mask(self, radius, r):
        """ Obtains Radial High and Low Pass Filters
            Inputs:
                radius - radius of poalr grid
                r - specifies boundary where hi is 1 and where lo is mostly 1
            Outputs:
                lo_mask - Low Pass Filter 
                hi_mask - High Pass Filter
            """
        # shift log radius (shifts by an octave if log2(r) = 1)
        log_rad = np.log2(radius) - np.log2(r)
        
        hi_mask = np.clip(log_rad, -self.twidth, 0)
        hi_mask = np.abs(np.cos(hi_mask*np.pi/(2*self.twidth)))
        lo_mask = np.sqrt(1.0 - hi_mask**2)

        return lo_mask, hi_mask


    def _get_angle_mask(self, angle, b):
        """ Obtains Angle Mask 
            Inputs:
                angle - Angular component of polar coordinate grid
                b - bth band or orientation
            Outputs:
                angle_mask - Angular mask
            """
        order = self.nbands - 1
        const = np.power(2, (2*order)) * np.power(factorial(order), 2)/(self.nbands*factorial(2*order))
        angle = np.mod(np.pi + angle - np.pi*b/self.nbands, 2*np.pi) - np.pi

        if self.complex_pyr:
            # complex (single lobe)
            angle_mask = 2*np.sqrt(const) * np.power(np.cos(angle), order) * (np.abs(angle) < np.pi/2)
        else:
            # non-complex
            angle_mask = 2*np.sqrt(const) * np.power(np.cos(angle), order) 

        return angle_mask


    def get_filters(self, image):
        """ Obtains cropped? Steerable Pyramid Filters 
            Inputs: 
                image - input single channel image
            Outputs:
                filters - list of pyramid filters
                crops - crop indices associated with each filter
            """
        h, w = image.shape
        angle, radius = get_polar_grid(h, w)

        # rvals specify spacing between adjacent filters
        rvals = 2.0**np.arange(-4, 1)[::-1]

        # get initla Low and High Pass Filters
        lo_mask_prev, hi_mask = self._get_radial_mask(radius, r=rvals[0])

        # get initial crop index
        crop = get_filter_crops(hi_mask)

        filters = [hi_mask[crop[0]:crop[1], crop[2]:crop[3]]]
        crops = [crop]

        for idx, rval in enumerate(rvals[1:]):
            
            # obtain Radial Band Filter Mask
            lo_mask, hi_mask = self._get_radial_mask(radius, rval)
            rad_mask = hi_mask * lo_mask_prev

            # obtain crops indexes for current level
            if idx > 0:
                crop = get_filter_crops(rad_mask)

            # get filters at each band (orientation)
            for b in range(self.nbands):
                # get Anglular Filter Mask
                angle_mask = self._get_angle_mask(angle, b, self.nbands)
                
                filt = rad_mask*angle_mask/2
                filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]]) 

                # store crop dimensions for current Pyramid Level
                crops.append(crop)

            lo_mask_prev = lo_mask

        # get final Low Pass Filter Mask and crop dims
        crop = get_filter_crops(lo_mask)
        filters.append(lo_mask[crop[0]:crop[1], crop[2]:crop[3]])
        crops.append(crop)

        return filters, crops
    

    def build_pyramid(self, image, cropped_filters, crops, freq=False):
        """ Build Pyramid Decomposition
            Inputs:
                image - input single channel image
                cropped_filters - cropped filters
                crops - filter crop indices
            Outputs:
                pyramid - output list of pyramid decomposition
            """
        image_dft = np.fft.fftshift(np.fft.fft2(image))

        pyramid = []
        for filt, crop in zip(cropped_filters, crops):
            # get filtered/decomposed DFT 
            dft = image_dft[crop[0]:crop[1], crop[2]:crop[3]] * filt

            if freq:
                pyramid.append(dft)
            else:
                pyramid.append(np.fft.ifft2(np.fft.ifftshift(dft)))

        return pyramid
    

    def reconstruct_image_dft(self, pyramid, cropped_filters, crops, freq=False):
        """ Reconstructs image DFT from the pyramid decomposition.
            Inputs:
                pyramid - Complex Steerable Pyramid Decomposition 
                          (either spatial or frequency domain)
                cropped_filters - cropped filters
                crops - filter crop indices
                freq - flag to denote whether input pyramid is in frequency space 
            Outputs:
                recon_dft - reconstructed image DFT
            """
        h, w = pyramid[0].shape
        recon_dft = np.zeros((h, w), dtype=np.complex128)
        for i, (pyr, filt, crop) in enumerate(zip(pyramid, cropped_filters, crops)):
            # dft of sub band
            if freq:
                dft = pyr
            else:
                dft = np.fft.fftshift(np.fft.fft2(pyr))

            # accumulate reconstructed sub bands
            if self.complex_pyr and (i !=0 ) and (i != (len(cropped_filters) - 1)):
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += 2.0*dft*filt
            else:
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += dft*filt

        return recon_dft
    

    def reconstruct_image(self, pyramid, cropped_filters, crops, freq=False):
        """ Reconstructs image from the pyramid decomposition.
            Inputs:
                pyramid - Complex Steerable Pyramid Decomposition
                cropped_filters - cropped filters
                crops - filter crop indices
                freq - denotes whether input pyramid is in Frequency or Spatial Domain
            Outputs:
                recon_dft - reconstructed image DFT
            """
        recon_dft = self.reconstruct_image_dft(pyramid, cropped_filters, crops, freq)
        return np.fft.ifft2(np.fft.ifftshift(recon_dft)).real


    def display(self, filters, title=""):
        """ Displays all Pyramid Filters except for Hi and Lo pass masks 
            Inputs:
                filters - cropped filters list or pyramid list
                title - title for figure
            """
        fig, ax = plt.subplots(self.depth, self.nbands, figsize=(30, 20))
        fig.suptitle(title, size=22)

        idx = 0
        for i in range(self.depth):
            idx = i*self.nbands
            for j in range(1, self.nbands + 1):
                jdx = idx + j
                ax[i][j - 1].imshow(filters[jdx])

        plt.tight_layout();

        return fig, ax
    


## Sub Octave Smooth window Pyramid class
class SuboctaveSPR(SteerablePyramid):
    def __init__(self, depth, nbands, filters_per_octave, cos_order=6, complex_pyr=False):
        """ 
        depth - Pyramid Depth (height)
        nbands - number of orientations
        filters_per_octave = 6
        cos_order = 6
        complex_pyr - determines whether to create a Complex Pyramid

        NOTE: this class does not make full plots since it would take too long

        Also, there is an issue with non-complex pyramids, not sure what it is
        """
        self.depth = depth
        self.num_filts = depth*filters_per_octave
        self.nbands = nbands
        self.filters_per_octave = filters_per_octave
        self.cos_order = cos_order
        self.complex_pyr = complex_pyr


    def _get_radial_mask_smooth(self, radius):
        """ Obtains smooth radial filters and their square sum which is used 
            to compute lo and hi pass filters
            Inputs:
            Outputs:
            """
        # get log radius
        rad = np.log2(radius)
        rad = (self.depth + rad)/self.depth
        rad = rad*(np.pi/2 + np.pi/7*self.num_filts)

        h, w = radius.shape
        total = np.zeros((h, w))
        rad_filters = []
        const = np.power(2, 2*self.cos_order) \
                * np.power(factorial(self.cos_order), 2) \
                / ((self.cos_order + 1)*factorial(2*self.cos_order))
        
        for k in reversed(range(self.num_filts)):
            shift = np.pi/(self.cos_order+1)*(k+1)+2*np.pi/7
            rad_filters.append(np.sqrt(const)*np.power(np.cos(rad - shift), self.cos_order)*self.window_func(rad, shift))
            total += rad_filters[-1]**2

        return rad_filters, total
    

    def _get_angle_mask_smooth(self, angle, b):
        """ Obtains Angle Mask 
            Inputs:
                angle - Angular component of polar coordinate grid
                b - bth band or orientation
            Outputs:
                angle_mask - Angular mask
            """
        order = self.nbands - 1
        const = np.power(2, (2*order)) * np.power(factorial(order), 2)/(self.nbands*factorial(2*order))
        angle = np.mod(np.pi + angle - np.pi*b/self.nbands, 2*np.pi) - np.pi

        if self.complex_pyr:
            # complex (single lobe)
            angle_mask = np.sqrt(const) * np.power(np.cos(angle), order) * (np.abs(angle) < np.pi/2)
        else:
            # non-complex
            angle_mask = np.sqrt(const) * np.power(np.cos(angle), order) 

        return angle_mask


    @staticmethod
    def window_func(x, center):
        return np.abs(x - center) < np.pi/2


    def get_filters(self, image):
        """ Builds Filters 
            Inputs:
                image - input image
            Outputs:
                filters - list of pyramid filters
                crops - crop indices associated with each filter
            """
        h, w = image.shape
        angle, radius = get_polar_grid(h, w)

        # get log radius
        rad = np.log2(radius)
        rad = (self.depth + rad)/self.depth
        rad = rad*(np.pi/2 + np.pi/7*self.num_filts)
        
        # Build Radial Filters
        rad_filters = []
        total = np.zeros((h, w))
        const = np.power(2, 2*self.cos_order) \
                * np.power(factorial(self.cos_order), 2) \
                / ((self.cos_order + 1)*factorial(2*self.cos_order))
        
        for k in reversed(range(self.num_filts)):
            shift = np.pi/(self.cos_order+1)*(k+1)+2*np.pi/7
            rad_filters.append(np.sqrt(const)*np.power(np.cos(rad - shift), self.cos_order)*self.window_func(rad, shift))
            total += rad_filters[-1]**2


        # get lo and hi pass filters
        dims = np.array([h, w])

        center = np.ceil(dims/2).astype(int)
        lodims = np.ceil(center/4).astype(int)

        idx11 = center[0] - lodims[0]
        idx12 = center[0] + lodims[0]
        idx21 = center[1] - lodims[1]
        idx22 = center[1] + lodims[1]

        total_crop = total[idx11:idx12, idx21:idx22]

        lopass = np.zeros((h, w))
        lopass[idx11:idx12, idx21:idx22] = np.abs(np.sqrt(1 - total_crop + 1e-8))

        hipass = np.abs(np.sqrt(1 - (total + lopass**2) + 1e-8));

        # buils angle masks
        angle_masks = []
        for b in range(self.nbands):
            angle_masks.append(self._get_angle_mask_smooth(angle, b))


        # Get Sub Band Filters and Crops
        filters = []
        crops = []

        filters = [hipass]
        crops.append(get_filter_crops(hipass))

        for rad_filt in rad_filters:
            for ang_mask in angle_masks:
                filt = rad_filt*ang_mask
                crop = get_filter_crops(filt)
                
                filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])
                crops.append(crop)

        crop = get_filter_crops(lopass)
        filters.append(lopass[crop[0]:crop[1], crop[2]:crop[3]])  
        crops.append(crop)

        return filters, crops
    
    










# old code


    # def get_filters(self, image):
    #     """ Obtains Complex Steerable Pyramid Filters 
    #         Inputs: 
    #             image - input single channel image
    #         Outputs:
    #             filters - list of pyramid filters
    #             crops - crop indices associated with each filter
    #         """
    #     h, w = image.shape
    #     angle, radius = get_polar_grid(h, w)

    #     # rvals specify spacing between adjacent filters
    #     rvals = 2.0**np.arange(-self.depth, 1)[::-1]

    #     # get initla Low and High Pass Filters
    #     lo_mask_prev, hi_mask = self._get_radial_mask(radius, r=rvals[0])

    #     # get initial crop index
    #     crop_idx = get_filter_crops(hi_mask)

    #     filters = [hi_mask]
    #     crops = [crop_idx]

    #     for idx, rval in enumerate(rvals[1:]):
            
    #         # obtain Radial Band Filter Mask
    #         lo_mask, hi_mask = self._get_radial_mask(radius, rval)
    #         rad_mask = hi_mask * lo_mask_prev

    #         # obtain crops indexes for current level
    #         if idx > 0:
    #             crop_idx = get_filter_crops(rad_mask)

    #         # get filters at each band (orientation)
    #         for b in range(self.nbands):
    #             # get Anglular Filter Mask
    #             angle_mask = self._get_angle_mask(angle, b, self.nbands)
                
    #             filt = rad_mask*angle_mask/2
    #             filters.append(filt) 

    #             # store crop dimensions for current Pyramid Level
    #             crops.append(crop_idx)

    #         lo_mask_prev = lo_mask

    #     # get final Low Pass Filter Mask and crop dims
    #     filters.append(lo_mask)
    #     crops.append(get_filter_crops(lo_mask))

    #     return filters, crops