import numpy as np
import cv2


def convert_to_numpy(image, depth, label, depth_norm=10):
        """ converts image, depth, and label into usable numpy format """

        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        depth = depth.detach().cpu().squeeze().numpy()
        label = label.detach().cpu().squeeze().numpy()

        """ undo imagenet normalization and convert to uint8
                y = (x - mu)/std
                x = y*std + mu
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = cv2.normalize((image * std) + mean, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC3)
        
        # undo depth normalization
        depth *= depth_norm

        return image, depth, label
