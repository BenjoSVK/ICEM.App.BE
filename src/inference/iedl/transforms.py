import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import joblib
from pathlib import Path
from typing import Callable


class NumpyToTensor:
    def __call__(self, x):
        x = x.astype(np.uint8)
        x = np.transpose(x, axes=(2, 0, 1))
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)
        x = (x / 255.0).float()
        return x


class DistanceTransform:
    def __call__(self, x):
        """Converts <4;H;W> where first 3 channels are RGB and last is cell mask
        Normalized RGB <0;1> and 3 distance maps for each class
        """
        y = np.zeros((6, x.shape[1], x.shape[2]), dtype=np.float32)

        # First 3 channels are RGB converted into LAB
        y[:3] = x[:3].astype(np.float32) / 255.0
        y[:3] = cv2.cvtColor(y[:3].transpose(1, 2, 0), cv2.COLOR_RGB2LAB).transpose(2, 0, 1)

        # Next 3 channels are distance transforms for each class
        d1 = (x[3] == 1).astype(np.uint8)
        d2 = (x[3] == 2).astype(np.uint8)
        d3 = (x[3] == 3).astype(np.uint8)
        y[3] = cv2.distanceTransform(d1, cv2.DIST_L2, 3).astype(np.float32)
        y[4] = cv2.distanceTransform(d2, cv2.DIST_L2, 3).astype(np.float32)
        y[5] = cv2.distanceTransform(d3, cv2.DIST_L2, 3).astype(np.float32)

        return y

        

class MultiScale:
    def __init__(self, scalers_path: Path, downscaled: bool = True):
        self.downscaled = downscaled
        self.scalers = []
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/L_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/A_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/B_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst1_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst2_scaler.joblib"))
        self.scalers.append(joblib.load(f"{scalers_path}/multilabel/multiclass_dst3_scaler.joblib"))

    def __call__(self, x):
        if self.downscaled:
            x = x.transpose(1, 2, 0)
            x = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2))
            x = x.transpose(2, 0, 1)

        H, W = x.shape[1], x.shape[2]
        for i, scaler in enumerate(self.scalers):
            x[i] = scaler.transform(x[i].reshape(-1, 1)).reshape(H, W)

        x = np.expand_dims(x, axis=0)
        return x