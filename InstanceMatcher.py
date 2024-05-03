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

        
class InstanceMatcher:
    def __init__(self, model_folder):
        self.sift = cv.SIFT_create()
        models = []
        for filename in glob.glob(f"{model_folder}/*.png"):
            model_img = cv.imread(filename)
            # OFFLINE PHASE
            features, descriptors = self.compute_features(model_img)
            models.append((StarModel(features, descriptors), filename, model_img.shape[:2]))
        self.models = models
        
    def sharpen(self, img, k, sigma):
        m = (2 * int(3 * sigma) + 1)
        g = cv.getGaussianKernel(m, sigma)
        F_b = np.dot(g, g.T)
        F_id = np.zeros((m, m)); 
        F_id[m // 2, m // 2] = 1
        sharpen_filter = F_id + k * (F_id - F_b)
        return cv.filter2D(img, -1, sharpen_filter)
            
    def preprocess(self, img):
        img = cv.medianBlur(img, 5) # impulse noise
        img = cv.fastNlMeansDenoisingColored(img, None) # gaussian noise
        img = self.sharpen(img, 1, 1) # sharpen
        return img
    
    def compute_features(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        features = [Feature(kp, des) for kp, des, in zip(keypoints, descriptors)]
        return features, descriptors

    def match(self, target_img, quantization_step, th="auto"):
        target_img = self.preprocess(target_img) 
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
            
            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(cv.imread(filename)[:,:,::-1])
            # axes[0].set_axis_off()
            # axes[1].imshow(aa.arr, cmap='jet', interpolation='nearest')
            # axes[1].set_axis_off()
            # for i in range(aa.arr.shape[0]):
            #     for j in range(aa.arr.shape[1]):
            #         axes[1].text(j, i, f"{aa.arr[i, j]:.0f}", ha="center", va="center", color="w")
            # plt.suptitle(filename)
            # plt.tight_layout()
            # plt.show()
        
        if th == "auto":
            th = np.mean(np.concatenate([aa.get_votes() for aa, _, _ in found]))
        
        for aa, filename, model_img_shape in found:
            for model_pts, target_pts in aa.get_matching_pts(th):
                
                model_pts = np.float32(model_pts).reshape((-1,1,2))       
                target_pts = np.float32(target_pts).reshape((-1,1,2))
                
                H, _ = cv.findHomography(model_pts, target_pts, cv.RANSAC, 5.0)
                
                h, w = model_img_shape
                pts = np.float32([ [0, 0],[0, h - 1],[w - 1, h - 1],[w - 1, 0] ]).reshape(-1,1,2)
                
                dst = cv.perspectiveTransform(pts, H)

                target_img = cv.polylines(target_img,[np.int32(dst)],True, (0, 255, 0), 20, cv.LINE_AA)

        plt.imshow(target_img[:,:,::-1])
        plt.axis('off')
        plt.show()