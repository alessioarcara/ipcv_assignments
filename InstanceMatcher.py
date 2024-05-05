import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from Accumulator import Accumulator

LOWE_THRESHOLD = 0.7


class Feature:
    def __init__(self, kp, des):
        self.pt = kp.pt
        self.angle = kp.angle 
        self.scale = kp.size
        
    
class StarModel:
    def __init__(self, features, descriptors):
        self.features = features
        self.descriptors = descriptors
        
        x = 0
        y = 0
        n = len(features)
        
        for f in features:
            x += f.pt[0]
            y += f.pt[1]
            
        self.barycenter = (x/n, y/n)
        self.joining_vectors = [(self.barycenter[0] - f.pt[0], self.barycenter[1] - f.pt[1]) for f in features]
        

class BoundingBox:
    """
    pt1: The top-left point.
    pt2: The bottom-right point.
    """
    def __init__(self, pts):
        self.pt1 = pts[0]
        self.pt2 = pts[1]
        
    def get_pos(self):
        return ((self.pt1[0] + self.pt2[0]) / 2, (self.pt1[1] + self.pt2[1]) / 2)
    
    def get_size(self):
        return (self.pt2[0] - self.pt1[0], self.pt2[1] - self.pt1[1])
    
    def get_pt1(self):
        return tuple(map(int, self.pt1))
    
    def get_pt2(self):
        return tuple(map(int, self.pt2))
        
class InstanceMatcher:
    def __init__(self, model_folder, preprocessing_steps=[]):
        self.sift = cv.SIFT_create()
        self.models = self.load_models(model_folder)
        self.preprocessing_steps = preprocessing_steps
        
    def load_models(self, model_folder):
        models = []
        for filename in glob.glob(f"{model_folder}/*.png"):
            model_img = cv.imread(filename)
            # OFFLINE PHASE
            features, descriptors = self.compute_features(model_img)
            models.append((StarModel(features, descriptors), filename, model_img.shape[:2]))
        return models
            
    def preprocess(self, img):
        for preprocess in self.preprocessing_steps:
            img = preprocess(img)
        return img
    
    def compute_features(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        features = [Feature(kp, des) for kp, des, in zip(keypoints, descriptors)]
        return features, descriptors

    def match(self, target_img, quantization_step, th=None):
        target_img = self.preprocess(cv.cvtColor(cv.imread(target_img), cv.COLOR_BGR2RGB))
        # ONLINE PHASE
        target_features, target_descriptors = self.compute_features(target_img)

        found = []
        for (model, filename, model_img_shape) in self.models:
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            matches = matcher.knnMatch(model.descriptors, target_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < LOWE_THRESHOLD * n.distance]
            
            aa = Accumulator(target_img.shape, quantization_step)
            for m in good_matches:
                p_j = target_features[m.trainIdx].pt
                p_i = model.features[m.queryIdx].pt
                v_i = model.joining_vectors[m.queryIdx]
                
                ds = target_features[m.trainIdx].scale / model.features[m.queryIdx].scale
                da = target_features[m.trainIdx].angle - model.features[m.queryIdx].angle
                theta = np.radians(da)
                R = np.array([[np.cos(theta), -np.sin(theta)], 
                              [np.sin(theta), np.cos(theta)]])
              
                v_i = ds * np.dot(R, v_i)
                
                barycenter_j = (p_j[0] + v_i[0], p_j[1] + v_i[1])
                aa.cast_vote(barycenter_j, (p_i, p_j))
            
            aa.nms()
            if aa.check_votes():
                found.append((aa, filename, model_img_shape))
           
            # if show_accumulator:                
            #     fig, axes = plt.subplots(1, 2)
            #     axes[0].imshow(cv.imread(filename)[:,:,::-1])
            #     axes[0].set_axis_off()
            #     axes[1].imshow(aa.arr, cmap='jet', interpolation='nearest')
            #     axes[1].set_axis_off()
            #     for i in range(aa.arr.shape[0]):
            #         for j in range(aa.arr.shape[1]):
            #             axes[1].text(j, i, f"{aa.arr[i, j]:.0f}", ha="center", va="center", color="w")
            #     plt.suptitle(filename)
            #     plt.tight_layout()
            #     plt.show()
        
        if th == None:
            try:
                th = np.mean(np.concatenate([aa.get_votes() for aa, _, _ in found]))
            except:
                return [], []

        instances = []
        boxes = []
        for aa, filename, model_img_shape in found:
            for model_pts, target_pts in aa.get_matching_pts(th):
                                
                model_pts = np.float32(model_pts).reshape((-1,1,2))       
                target_pts = np.float32(target_pts).reshape((-1,1,2))
                H, _ = cv.findHomography(model_pts, target_pts, cv.RANSAC, 5.0)
                pts = np.float32([[0, 0],[model_img_shape[1]-1, model_img_shape[0]-1]]).reshape(-1,1,2)
                transformed_pts = cv.perspectiveTransform(pts, H).reshape(-1, 2)
                
                instances.append(filename)
                boxes.append(BoundingBox(transformed_pts))

        return instances, boxes