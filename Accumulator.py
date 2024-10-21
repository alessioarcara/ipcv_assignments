import numpy as np
import cv2 as cv


class Accumulator:
    def __init__(self, shape, quantization_step):
        self.arr = np.zeros((shape[0] // quantization_step, shape[1] // quantization_step))
        self.quantization_step = quantization_step
        self.matches = {}
        
    def cast_vote(self, pt, matching_pts):
        x, y = int(pt[0] / self.quantization_step), int(pt[1] / self.quantization_step)
        if 0 <= x < self.arr.shape[1] and 0 <= y < self.arr.shape[0]:
            self.arr[y, x] += 1
            if (y, x) in self.matches:
                self.matches[(y, x)].append(matching_pts)
            else: 
                self.matches[(y, x)] = [matching_pts]
                
    def check_votes(self):
        return np.sum(self.arr) > 0
    
    def get_votes(self):
        return self.arr[self.arr > 0]
    
    def get_matching_pts(self, th):
        x_idxs, y_idxs = np.where(self.arr >= th)
        
        matching_pts = []
        
        for x, y in zip(x_idxs, y_idxs):
            model_pts = [model_pt for model_pt, _ in self.matches[x, y]]
            target_pts = [target_pt for _, target_pt in self.matches[x, y]]
            matching_pts.append((model_pts, target_pts))
            
        return matching_pts

    def nms(self, size=3, min_votes=4):
        b = size // 2
        padded_arr = cv.copyMakeBorder(self.arr, b, b, b, b, cv.BORDER_CONSTANT, value=0)
        nms_arr = np.zeros(self.arr.shape)
        
        for i in range(b, self.arr.shape[0] + b):
            for j in range(b, self.arr.shape[1] + b):
                val = padded_arr[i, j]
                window = padded_arr[i - b:i + b + 1, j - b:j + b + 1]
                if val == np.max(window) and val >= min_votes:
                    nms_arr[i - b, j - b] = val
                    
        self.arr = nms_arr