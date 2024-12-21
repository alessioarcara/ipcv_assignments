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

    def nms_3d(self, min_votes, size=3):
        h, w, _ = self.shape
        b = size // 2

        padded_array = np.pad(self.arr, pad_width=((b, b), (b, b), (0, 0)), mode='constant', constant_values=0)
        nms_result = np.zeros((h, w))
        filtered_matches = {}

        for y in range(h):
            for x in range(w):
                window = padded_array[y:y + size, x:x + size, :]
                central_values = padded_array[y + b, x + b, :]
                
                max_value = np.max(window)

                if np.any(central_values == max_value) and max_value >= min_votes:
                    nms_result[y, x] = max_value
                    model_idx = np.argmax(central_values)
                    filtered_matches[(y, x, model_idx)] = self.matches[(y, x, model_idx)]

        return nms_result, filtered_matches