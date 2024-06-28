import numpy as np
import pandas as pd


def calc_TPR_and_FPR(predictions, targets):
    
    map_individuals = {0:3, 1:5, 2:7, 3:9, 4:10, 5:17, 6:19, 7:20}
    
    class_count = len(map_individuals)
    TP = {3:0, 5:0, 7:0, 9:0, 10:0, 17:0, 19:0, 20:0}
    FP = {3:0, 5:0, 7:0, 9:0, 10:0, 17:0, 19:0, 20:0}
    TN = {3:0, 5:0, 7:0, 9:0, 10:0, 17:0, 19:0, 20:0}
    FN = {3:0, 5:0, 7:0, 9:0, 10:0, 17:0, 19:0, 20:0}
    
    for class_index in map_individuals.values():
        for i in range(len(predictions)):
            pred = predictions[i]
            target = targets[i]
            if pred == class_index and target == class_index:
                TP[class_index] += 1
            elif pred == class_index and target != class_index:
                FP[class_index] += 1
            elif pred != class_index and target == class_index:
                FN[class_index] += 1
            elif pred != class_index and target != class_index:
                TN[class_index] += 1
    
    TPR = [TP[j] / (TP[j] + FN[j]) if (TP[j] + FN[j]) != 0 else 0 for j in map_individuals.values()]
    FPR = [FP[j] / (FP[j] + TN[j]) if (FP[j] + TN[j]) != 0 else 0 for j in map_individuals.values()]
    
    return TPR, FPR
    