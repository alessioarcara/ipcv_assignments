import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
from Accumulator import Accumulator


class Feature:
    def __init__(self, kp):
        self.pt = kp.pt
        self.angle = kp.angle 
        self.scale = kp.size
        
    
class StarModel:
    def __init__(
            self, 
            features, 
            descriptors, 
            model_shape
        ):
        self.features = features
        self.descriptors = descriptors
        self.model_shape = model_shape
        
        pts = np.array([f.pt for f in features])
        self.barycenter = np.mean(pts, axis=0)
            
        self.joining_vectors = [
            (self.barycenter[0] - f.pt[0], self.barycenter[1] - f.pt[1]) 
            for f in features
        ]


class BoundingBox:
    """
    pt1: The top-left point.
    pt2: The bottom-right point.
    """
    def __init__(self, pts, model_name, confidence):
        self.pt1 = tuple(map(int, pts[0]))
        self.pt2 = tuple(map(int, pts[1]))
        self.model_name = model_name
        self.confidence = confidence
        
    def center(self):
        return (self.pt1[0] + self.pt2[0]) / 2, (self.pt1[1] + self.pt2[1]) / 2
    
    def width(self):
        return self.pt2[0] - self.pt1[0]
        
    def height(self):
        return self.pt2[1] - self.pt1[1]
    
    def get_pts(self):
        return self.pt1, self.pt2


class InstanceMatcher:
    LOWE_THRESHOLD  = 0.75

    def __init__(
        self, 
        model_folder, 
        preprocessing_steps=None, 
        feature_detector=None,
        show_models = False 
    ):
        self.preprocessing_steps = preprocessing_steps or []
        self.feature_detector = feature_detector or cv.SIFT_create()
        self.models, self.file_loading_order = self._load_models(model_folder)
        if show_models:
            pass
        print(f"Loaded {len(self.models)} models from {model_folder}")
        
    def _load_models(self, model_folder):
        """
        GHT Offline Phase 
        """
        models = []
        file_loading_order = {}
        for idx, filename in enumerate(glob.glob(f"{model_folder}/*.png")):
            model_img = cv.imread(filename)
            if model_img is None:
                print(f"Couldn't read image: {filename}")
                continue

            features, descriptors = self._compute_features(model_img)
            models.append(StarModel(features, descriptors, model_img.shape[:2]))
            file_loading_order[idx] = filename
        return models, file_loading_order 
            
    def preprocess(self, img):
        for preprocess in self.preprocessing_steps:
            img = preprocess(img)
        return img
    
    def _compute_features(self, img):
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        features = [Feature(kp) for kp in keypoints]
        return features, descriptors

    def _extract_bboxes(self, found_matches): 
        bboxes = []

        for (_, _, model_idx), value in found_matches.items():
            model_pts = [model_pt for model_pt, _ in value]
            target_pts = [target_pt for _, target_pt in value]
            vote_count = len(value)

            model_pts = np.float32(model_pts).reshape((-1,1,2))       
            target_pts = np.float32(target_pts).reshape((-1,1,2))

            H, _ = cv.findHomography(model_pts, target_pts, cv.RANSAC, 5.0)
                
            pts = np.float32([[0, 0],[self.models[model_idx].model_shape[1]-1, self.models[model_idx].model_shape[0]-1]]).reshape(-1,1,2)
            transformed_pts = cv.perspectiveTransform(pts, H).reshape(-1, 2)
                
            bboxes.append(BoundingBox(transformed_pts, self.file_loading_order[model_idx], vote_count))

        return bboxes
    
    def match(
        self, 
        target_img_path, 
        quantization_step, 
        nms_th=4,
        nms_size=5, 
        show_accumulator=False
    ):
        """
        GHT Online Phase 
        """
        target_img = self.preprocess(cv.cvtColor(cv.imread(target_img_path), cv.COLOR_BGR2RGB))
        target_features, target_descriptors = self._compute_features(target_img)

        aa = Accumulator(target_img.shape, len(self.models), quantization_step)

        for i, model in enumerate(self.models):
            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            matches = matcher.knnMatch(target_descriptors, model.descriptors, k=2)

            good_matches = [m for m, n in matches if m.distance < self.LOWE_THRESHOLD * n.distance]
            
            for m in good_matches:
                target_feature = target_features[m.queryIdx]
                model_feature = model.features[m.trainIdx]

                p_j = target_feature.pt
                p_i = model_feature.pt
                v_i = model.joining_vectors[m.trainIdx]

                ds = target_feature.scale / model_feature.scale
                da = (target_feature.angle - model_feature.angle) % 360

                theta = np.radians(da)
                R = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
              
                v_i_transformed = ds * np.dot(R, v_i)
                
                barycenter_j = (
                    p_j[0] + v_i_transformed[0], 
                    p_j[1] + v_i_transformed[1]
                )
                
                aa.cast_vote(i, barycenter_j, (p_i, p_j))
            
        aa_2d, found_matches = aa.nms_3d(min_votes=nms_th, size=nms_size)

        if show_accumulator:
            InstanceMatcher.display_accumulator(aa_2d, target_img_path.split("/")[-1])
        
        return self._extract_bboxes(found_matches)
    
    @staticmethod
    def display_3d_accumulator(accumulator):
        pass

    
    @staticmethod
    def display_accumulator(accumulator, target_img_path=""):
        plt.imshow(accumulator, cmap='jet', interpolation='nearest')
        plt.axis('off')
        plt.title(f"Accumulator Heatmap for the scene: {target_img_path}")

        for y in range(accumulator.shape[0]):
            for x in range(accumulator.shape[1]):
                plt.text(
                    x, y, f"{accumulator[y, x]:.0f}", 
                    ha="center", va="center", color="white"
                )
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_models():
        pass