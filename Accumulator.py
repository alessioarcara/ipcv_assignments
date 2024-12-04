import numpy as np

class Accumulator:
    """
    # 3D Accumulator with spatial dimensions matching the target image (quantized with a step) and depth equal the number of models
    """
    def __init__(self, target_img_shape, n_models, quantization_step):
        self.quantization_step = quantization_step
        self.shape = (
            target_img_shape[0] // quantization_step,
            target_img_shape[1] // quantization_step, 
            n_models
        )
        self.arr = np.zeros(self.shape)
        self.matches = {}
        
    def cast_vote(self, model_idx, pt, matching_pts):
        x, y = int(pt[0] / self.quantization_step), int(pt[1] / self.quantization_step)
        if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
            self.arr[y, x, model_idx] += 1
            self.matches.setdefault((y, x, model_idx), []).append(matching_pts)

    def get_matching_pts(self, th=1):
        y_idxs, x_idxs = np.where(self.arr >= th)
        matching_pts = [
        (
            self.arr[y, x],  # Numero di voti nel punto (y, x).
            (
                [model_pt for model_pt, _ in self.matches[(y, x)]],
                [target_pt for _, target_pt in self.matches[(y, x)]],
            )
        )
            for y, x in zip(y_idxs, x_idxs)
        ]
        return matching_pts

    def nms_3d(self, min_votes, size=3):
        h, w, _ = self.shape
        b = size // 2
        
        padded_arrays = np.pad(self.arr, pad_width=((0, 0), (b, b), (b, b)), mode='constant', constant_values=0)
        nms_result = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                window = padded_arrays[:, y:y + size, x:x + size]
                central_values = padded_arrays[:, y + b, x + b]

                max_value = np.max(window)

                if np.any(central_values == max_value) and max_value >= min_votes:
                    nms_result[y, x] = max_value

        return nms_result