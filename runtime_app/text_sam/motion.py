"""Propagate the preceding mask with backward optical flow between SAM updates."""
import cv2
import numpy as np


class MotionPropagation:
    def __init__(self):
        self.previous = None
        self.grid = None
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def gray(self, rgb):
        h, w = rgb.shape[:2]
        width = min(320, w)
        return cv2.cvtColor(cv2.resize(rgb, (width, max(16, round(h*width/w)))), cv2.COLOR_RGB2GRAY)

    def propagate(self, rgb, masks):
        current = self.gray(rgb)
        if self.previous is None or not masks:
            return {}
        h, w = rgb.shape[:2]
        if self.grid is None or self.grid[0].shape != (h,w):
            yy, xx = np.mgrid[:h, :w]
            self.grid = xx.astype(np.float32), yy.astype(np.float32)
        # remap needs current -> preceding coordinates, so estimate backward flow.
        flow = self.flow.calc(current, self.previous, None)
        fh, fw = flow.shape[:2]
        flow = cv2.resize(flow, (w,h))
        map_x = self.grid[0] + flow[:,:,0] * (w/fw)
        map_y = self.grid[1] + flow[:,:,1] * (h/fh)
        return {i: cv2.remap(mask.view(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).view(bool) for i,mask in masks.items()}

    def remember(self, rgb):
        self.previous = self.gray(rgb)
