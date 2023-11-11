import numpy as np
import cv2
import torch
import torch.nn.functional as F
from raft import RAFT
from utils import flow_viz
# from utils.utils import InputPadder


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 
        Obtained form RAFT dataset
        """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    

def process_img(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def load_model(weights_path, args):
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cuda")
    return model


def inference(model, frame1, frame2, device, pad_mode='sintel',
              iters=12, flow_init=None, upsample=True, test_mode=True):

    model.eval()
    with torch.no_grad():
        # preprocess
        frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()[None].to(device)
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()[None].to(device)

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow
        if test_mode:
          flow_low, flow_up = model(frame1,
                                    frame2,
                                    iters=iters,
                                    flow_init=flow_init,
                                    upsample=upsample,
                                    test_mode=test_mode)
          return flow_low, flow_up

        else:
            flow_iters = model(frame1,
                               frame2,
                               iters=iters,
                               flow_init=flow_init,
                               upsample=upsample,
                               test_mode=test_mode)

            return flow_iters



# sketchy class to pass to RAFT
class Args():
  def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
    self.model = model
    self.path = path
    self.small = small
    self.mixed_precision = mixed_precision
    self.alternate_corr = alternate_corr

  """ Sketchy hack to pretend to iterate through the class objects """
  def __iter__(self):
    return self

  def __next__(self):
    raise StopIteration
  

def viz(_frame, y_true, y_pred):
    """ Draws true and predicted speeds on a frame """
    # ensure that frame is a numpy array and copy so we don't overwrite it
    if isinstance(_frame, torch.Tensor):
        frame = _frame.numpy().transpose((1, 2, 0)).copy() 
        frame = cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    else:
        frame = _frame.copy()
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org_true = (15, 30)
    org_pred = (15, 65)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color_true = (5, 255, 5)
    color_pred = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 3

    # draw speeds on frame
    frame = cv2.putText(frame, f"True Speed: {y_true:.3f}", org_true, font,
                fontScale, (0,0,0), 15, cv2.LINE_AA)
    frame = cv2.putText(frame, f"True Speed: {y_true:.3f}", org_true, font,
                        fontScale, color_true, thickness, cv2.LINE_AA)

    frame = cv2.putText(frame, f"Pred Speed: {y_pred:.3f}", org_pred, font,
                        fontScale, (0,0,0), 15, cv2.LINE_AA)
    frame = cv2.putText(frame, f"Pred Speed: {y_pred:.3f}", org_pred, font,
                        fontScale, color_pred, thickness, cv2.LINE_AA)
    
    return frame

