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


import sklearn
print(sklearn.__version__)


sentence_token_level = {
    "max_size_aux_verbs": None,
    "max_size_passive": None,
    "max_size_named_entities": None,
    "max_size_np_pp_modifiers": None,
    "max_size_subordination": None,
    "max_size_coordination": None,
    }
document_token_level = {
    "total_token_ratio_aux_verbs": None,
    "total_token_ratio_passive": None,
    "total_token_ratio_named_entities": None,
    "total_token_ratio_subordination": None,
    "ratio_clitic_per_token": None,
    "ratio_post_clitic_pronouns_per_token": None,
    "ratio_pre_clitic_pronouns_per_token": None,
    "ratio_named_entities_per_token": None,
    "ratio_aux_verbs_per_token": None,
    "ratio_subordination_per_token": None,
    "ratio_passive_per_token": None,
    "sophisticated_ratio": None,
    "concrete_ratio": None,
    "hapax_legomena_lemma_ratio": None,
    # "p0-p75_freq_ratio": None, missing from output
    # "p0-p75_freq_lemma_ratio": None, missing from output
    "ratio_aux_verbs_per_verb": None,
    "ratio_subordination_per_verb": None,
    "ratio_passive_per_verb": None,
    "ratio_coordination_per_token": None,
    "total_token_ratio_coordination": None,
    "ratio_coordination_per_token": None,
}

sentence_level= {
      "parse_depth": None,
      "words_after_verb": None,
      "words_before_verb": None,
      "sentence_length": None
    }

document_document_level = {
    "word_count": None,
    "sentence_count": None,
    "lexical_diversity": None}

token_level_high = {
      "word_length": None,
      "word_syllables": None,
      "ortho_neighbors": None,
      "ortho_neighbors_+freq_cum": None,
      "age_of_acquisition": None,
      "complexity": None,
    }

token_level_low = {
      "familiarity": None,
      "lexical_frequency": None
    }

thresholds_init = {"sentence-token-level": copy.deepcopy(sentence_token_level), "document-token-level": copy.deepcopy(document_token_level), "sentence-level": copy.deepcopy(sentence_level), "document-document-level":copy.deepcopy(document_document_level), "token-level-high": copy.deepcopy(token_level_high), "token-level-low": copy.deepcopy(token_level_low)}

classes = {'Très Facile':'N1', 'Facile': 'N2', 'Accessible':'N3','+Complexe':'N4'}

thresholds = {'N1':copy.deepcopy(thresholds_init), 'N2':copy.deepcopy(thresholds_init), 'N3': copy.deepcopy(thresholds_init), 'N4':copy.deepcopy(thresholds_init)}
densities = {'N1':copy.deepcopy(thresholds_init), 'N2':copy.deepcopy(thresholds_init), 'N3': copy.deepcopy(thresholds_init), 'N4':copy.deepcopy(thresholds_init)}

distrib_levels = {"sentence-token-level": "document", "document-token-level": "document", "sentence-level": "sentence", "document-document-level": "document", "token-level-high": "token", "token-level-low": "token"}

classes = ['N1', 'N2', 'N3', 'N4']


def get_prob_threshold_f1(y_labels, y_pred_prob):
    precision, recall, thresholds = precision_recall_curve(y_labels, y_pred_prob)
    #print(precision, recall)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0
    #print(f1_scores[f1_scores<0])
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold


def get_input_threshold(x_values, y_labels):
    # Example data (inputs and labels)
    x_values = np.array(x_values)  # input values (single feature, reshaped)
    y_labels = np.array(y_labels)  # labels (0 or 1)

    # Create a Logistic Regression model
    model = LogisticRegression(class_weight='balanced')#, solver='lbfgs', penalty=None)

    # Train the model on the data
    model.fit(x_values, y_labels)

    # Make predictions (probability and class prediction)
    y_pred_prob = model.predict_proba(x_values)[:, 1]  # Probability for class 1
    y_pred_class = model.predict(x_values)  # Predicted class labels

    acc = accuracy_score(y_labels, y_pred_class)
    print(f"Training accuracy: {acc:.3f}")

    # Decision boundary (where probability is 0.5)
    # plt.axvline(x=model.intercept_ / -model.coef_, color='green', linestyle='--', label='Decision Boundary')

    # Desired threshold
    threshold = 0.5
    # Get weight and bias from model
    w = model.coef_[0][0]
    b = model.intercept_[0]
    # Compute x for desired threshold
    x_thresh = -(np.log(1 / threshold - 1) + b) / w
    # print(model.intercept_ , -model.coef_)
    # x_thresh = (model.intercept_ / -model.coef_).item()
    if x_thresh == np.nan or np.isnan(x_thresh):
        print("COEF", w)
        #print(x_values)
    return x_thresh


def compute_thresholds(thresholds, distributions):

    for classe, dico in distributions.items():
        if classe == "N4": break
        print('CLASSE:', classe)
        for level, dico2 in dico.items():
            for heuristic, values in dico2.items():
                x_values = []
                y_labels = []
                # if heuristic != 'words_after_verb': continue
                for c in classes:
                    values = distributions[c][level][heuristic]
                    x_values += [[v] for v in values]
                    if classes.index(c) > classes.index(classe):
                        y_labels += [1 for _ in range(len(values))]
                    else:
                        y_labels += [0 for _ in range(len(values))]

                print(classe, level, heuristic)
                thresh = get_input_threshold(x_values, y_labels)
                # print(round(thresh, 3))

                thresholds[classe][level][heuristic] = round(thresh, 3)

    return thresholds



if __name__ == '__main__':

    with open('./results/distributions.json') as json_data:
        distributions = json.load(json_data)

    thresholds = compute_thresholds(thresholds, distributions)

    with open('./results/thresholds_LogReg.json', 'w') as f:
        json.dump(thresholds, f)