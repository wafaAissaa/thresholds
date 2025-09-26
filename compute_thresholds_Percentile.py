import pandas as pd
import json
import os
from IPython.display import display, Markdown
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from compute_thresholds_IQR import get_bounds, sentence_token_level, document_token_level, sentence_level, document_token_level, token_level_high, token_level_low, thresholds_init, thresholds, densities, distrib_levels


import sklearn
print(sklearn.__version__)

classes = ['N1', 'N2', 'N3', 'N4']


def get_percentile(values, q):
    lower_bound = np.percentile(values, q)  # First quartile (25th percentile)
    upper_bound = np.percentile(values, 100-q)  # Third quartile (75th percentile)
    return lower_bound, upper_bound


def compute_thresholds(thresholds, distributions, q=5):

    for classe, dico in distributions.items():
        print('CLASSE:', classe)
        for level, dico2 in dico.items():
            for heuristic, values in dico2.items():
                lower_bound, upper_bound = get_percentile(values, q)
                print(classe, level, heuristic)
                if level == 'token-level-low':
                    thresholds[classe][level][heuristic] = lower_bound #round(lower_bound, 4)
                else:
                    thresholds[classe][level][heuristic] = upper_bound #round(upper_bound, 4)

    return thresholds



if __name__ == '__main__':

    for q in range(20, 100, 5):
        with open('./results/distributions.json') as json_data:
            distributions = json.load(json_data)

        thresholds = compute_thresholds(thresholds, distributions, q)
        with open('./results/thresholds_Percentile_%s.json' %q, 'w') as f:
            json.dump(thresholds, f)

