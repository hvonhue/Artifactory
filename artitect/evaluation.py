import pickle
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from artifact import Saw
from sliding_window_detector import SlidingWindowTransformerDetector, ConvolutionalSlidingWindowDetector, SlidingWindowLinearDetector
from mask_detector import WindowLinearDetector, WindowTransformerDetector, ConvolutionDetector

from data import RealisticArtifactDataset, CachedArtifactDataset, TestArtifactDataset, CenteredArtifactDataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix, fbeta_score

"""

"""
def detect_artifacts_sw(data_path: str, 
                        gt_path: str, 
                        threshold: float = 0.5, 
                        model_type: SlidingWindowTransformerDetector | ConvolutionalSlidingWindowDetector | SlidingWindowLinearDetector = SlidingWindowTransformerDetector,
                        model_path: str = "../models/SW_adaFCN_Trans.ckpt"):
    
    model = model_type.load_from_checkpoint(model_path).cpu()

    return None


def calculate_threshold_sw(val_path: str, 
                           model_type: SlidingWindowTransformerDetector | ConvolutionalSlidingWindowDetector | SlidingWindowLinearDetector = SlidingWindowTransformerDetector,
                           model_path: str = "../models/SW_adaFCN_Trans.ckpt"):
    
    detector = model_type.load_from_checkpoint(model_path).cpu()

    all_predictions_valSet = pd.DataFrame(columns=['Detector_name', 'predictions'])
    index = 0
    gt = list()
    val = CachedArtifactDataset(file=val_path)
    preds = list()

    # get all predictions
    for sample in val:
        example = sample["data"]
        window  = detector.window
        length  = len(example)

        # add artifact to data
        example_data = torch.tensor(example + sample["artifact"])

        # set detector to evaluation mode
        detector.eval()
        # make prediction and insert into prediction
        prediction = detector(example_data.unsqueeze(0))

        # update count
        preds = preds + [prediction.numpy()]

        gt = gt + [sample["label"]]

    # find best fbeta score
    for pred in preds:
        max_fbeta = 0

        for threshold in np.linspace(0,  1,  100):
            predictions = np.where(np.array(pred) > threshold, 1, 0)

            fbeta = fbeta_score(gt, predictions, average='macro', beta=0.5)

            if (fbeta > max_fbeta):
                max_fbeta = fbeta
                best_threshold_fbeta = threshold

            predictions = np.where(np.array(pred) > best_threshold_fbeta, 1, 0)

            tn, fp, fn, tp = confusion_matrix(gt, predictions, labels=[0, 1]).ravel()

            metric = pd.DataFrame([{
                'index': index,
                'detector': f"{model_path}",
                'threshold': best_threshold_fbeta,
                'fbeta_score': fbeta_score(gt, predictions, beta=0.5),
                'accuracy': accuracy_score(gt, predictions),
                'precision': precision_score(gt, predictions),
                'recall': recall_score(gt, predictions),
                'mse': mean_squared_error(gt, predictions), 
                'tn': tn,
                'fp': fp, 
                'fn': fn, 
                'tp': tp
            }])

        metrics = pd.concat([metrics, metric])

    return None


def detect_artifacts_mask(data_path: str, gt_path: str, model: WindowTransformerDetector):
    return None


def baseline_detector(input: torch.Tensor) -> int:   
    input.squeeze(0)
    prediction = 0

    center = int(input.shape[1]/2)
    # flag points with very high increment as artifact
    # Calculate increments by subtracting the tensor shifted by one from the original tensor
    increments = (input[0][1:] - input[0][:-1]).abs()
    mean_increment = torch.mean(increments)
    std_increment = torch.std(increments)

    if increments[center-1] > (mean_increment + 3*std_increment):
        prediction = 1
    
    return prediction