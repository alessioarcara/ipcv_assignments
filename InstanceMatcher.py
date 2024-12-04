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
            model_img_path, 
            model_shape
        ):
        self.features = features
        self.descriptors = descriptors
        self.model_img_path = model_img_path 
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
    
    def intersects(self, other):
        x_overlap = (self.pt1[0] < other.pt2[0]) and (self.pt2[0] > other.pt1[0])
        y_overlap = (self.pt1[1] < other.pt2[1]) and (self.pt2[1] > other.pt1[1])
        return x_overlap and y_overlap 

    @staticmethod
    def group_overlapping_boxes(bboxes):
        groups = []
        while bboxes:
            curr_bbox = bboxes.pop(0)
            group = [curr_bbox]

            for box in bboxes[:]:
                if curr_bbox.intersects(box):
                    group.append(box)
                    bboxes.remove(box)
            groups.append(group)

        return groups

        
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
        self.models = self._load_models(model_folder)
        if show_models:
            pass
        print(f"Loaded {len(self.models)} models from {model_folder}")
        
    def _load_models(self, model_folder):
        """
        GHT Offline Phase 
        """
        models = []
        for filename in glob.glob(f"{model_folder}/*.png"):
            model_img = cv.imread(filename)
            if model_img is None:
                print(f"Couldn't read image: {filename}")
                continue

            features, descriptors = self._compute_features(model_img)
            models.append(StarModel(features, descriptors, filename, model_img.shape[:2]))
        return models
            
    def preprocess(self, img):
        for preprocess in self.preprocessing_steps:
            img = preprocess(img)
        return img
    
    def _compute_features(self, img):
        keypoints, descriptors = self.feature_detector.detectAndCompute(img, None)
        features = [Feature(kp) for kp in keypoints]
        return features, descriptors

    def _nms_bboxes(self, bboxes):
        print(len(bboxes))
        groups = BoundingBox.group_overlapping_boxes(bboxes.copy())

        non_overlapping_boxes = []
        for group in groups:
            group.sort(key=lambda bbox: bbox.confidence, reverse=True)
            non_overlapping_boxes.append(group[0])
        return non_overlapping_boxes 

    def _extract_bboxes(self, found): 
        bboxes = []

        for aa, model_img_shape in found:
            for vote_count, (model_pts, target_pts) in aa.get_matching_pts():

                model_pts = np.float32(model_pts).reshape((-1,1,2))       
                target_pts = np.float32(target_pts).reshape((-1,1,2))

                H, _ = cv.findHomography(model_pts, target_pts, cv.RANSAC, 5.0)
                
                pts = np.float32([[0, 0],[model_img_shape[1]-1, model_img_shape[0]-1]]).reshape(-1,1,2)
                transformed_pts = cv.perspectiveTransform(pts, H).reshape(-1, 2)
                
                bboxes.append(BoundingBox(transformed_pts, aa.model_name, vote_count))

        return bboxes
    
    def match(
        self, 
        target_img_path, 
        quantization_step, 
        th=4, 
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
            
        if show_accumulator:
            pass
            # InstanceMatcher.display_accumulator(aa)

        aa.nms_3d(size=5, min_votes=th)
        return self._extract_bboxes(aa)
    
    @staticmethod
    def display_accumulator(accumulator, filename):
        pass
        # _, axes = plt.subplots(1, 2)
        # axes[0].imshow(cv.imread(filename)[:,:,::-1])
        # axes[0].set_axis_off()
        # axes[1].imshow(accumulator.arr, cmap='jet', interpolation='nearest')
        # axes[1].set_axis_off()
        # for i in range(accumulator.arr.shape[0]):
        #     for j in range(accumulator.arr.shape[1]):
        #         axes[1].text(j, i, f"{accumulator[i, j]:.0f}", ha="center", va="center", color="w")
        # plt.suptitle(filename)
        # plt.tight_layout()
        # plt.show()

    @staticmethod
    def display_models():
        pass