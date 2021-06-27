import os
import torch

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from simpletransformers.classification import (
    ClassificationModel
)
import logging
import imgkit
import sys

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix, hamming_loss
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, remove_stopwords
from nltk.corpus import stopwords
import argparse

import itertools
from sklearn.metrics import multilabel_confusion_matrix

from scipy.special import softmax
from sklearn.model_selection import KFold

from pathlib import Path
import logging
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import matplotlib.pyplot as plt
from highlight_text import HighlightText, ax_text, fig_text
import html, random
from IPython.core.display import display, HTML
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import word_tokenize
import re

from lime import lime_text
from lime.lime_text import LimeTextExplainer
import sys

# This is a path to a BERT model
# Our best model can be found at:
path_model = "best_model"
nltk.download('stopwords')
nltk.download('punkt')

def text_cleaning(txt, min_lenght=2):
    filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]

    words = preprocessing.preprocess_string(txt, filters)
    stop_words = set(stopwords.words('english'))
    stop_words.remove("no")
    stop_words.remove("than")
    stop_words.remove("not")

    c_words = [w for w in words if
               not w in stop_words and re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > min_lenght]

    out = ""
    out = ' '.join(map(str, c_words))

    return out


# Load Model
# train_args = {
#     'eval_batch_size': 32,
#     "n_gpu": 6,
#     'output_hidden_states': True,
#     'silent': True
# }
#
# model = ClassificationModel('bert', path_model, use_cuda=False)
# print("Model loaded.")
# model.model.config.output_hidden_states = True
# model.model.config.silent = True
#
#
# # LIME Predictor
# def predict(texts):
#     print("Called predict()")
#     results = []
#     for text in texts:
#         preds, raw_outputs, _, _ = model.predict([text])
#         probs = [softmax(prb) for prb in raw_outputs]
#         results.append(probs[0])
#     return np.array(results)
model = joblib.load("models/ps/model_logistic.pkl")
vectorizer = joblib.load("models/ps/modelvec_logistic.pkl")


def predict(texts):
    test_text = vectorizer.transform(texts)
    pred = model.predict_proba(test_text)
    print(pred)
    return pred


color_classes = {0: '65, 137, 225',  # blue
                 1: "234, 131, 4",  # orange
                 }


# function to normalize, if applicable
def normalize_MinMax(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def html_escape(text):
    return html.escape(text)


def highlight_full_data(lime_weights, data, pred):
    words_p = [x[0] for x in lime_weights if x[1] > 0]
    weights_p = np.asarray([x[1] for x in lime_weights if x[1] > 0])
    if len(weights_p) > 1:
        weights_p = normalize_MinMax(weights_p, t_min=min(weights_p), t_max=1)
    else:
        weights_p = [1]
    words_n = [x[0] for x in lime_weights if x[1] < 0]
    weights_n = np.asarray([x[1] for x in lime_weights if x[1] < 0])
    #     weights_n = normalize_MinMax(weights_n, t_min=max(weights_p), t_max=-0.8)

    if pred == 0:
        opposite = 1
    else:
        opposite = 0

    # positive values
    df_coeff = pd.DataFrame(
        {'word': words_p,
         'num_code': weights_p
         })
    word_to_coeff_mapping_p = {}
    for row in df_coeff.iterrows():
        row = row[1]
        word_to_coeff_mapping_p[row[0]] = row[1]

    # negative values
    df_coeff = pd.DataFrame(
        {'word': words_n,
         'num_code': weights_n
         })

    word_to_coeff_mapping_n = {}
    for row in df_coeff.iterrows():
        row = row[1]
        word_to_coeff_mapping_n[row[0]] = row[1]

    max_alpha = 1
    highlighted_text = []
    data = re.sub("-", " ", data)
    data = re.sub("/", "", data)
    for word in word_tokenize(data):
        if word.lower() in word_to_coeff_mapping_p or word.lower() in word_to_coeff_mapping_n:
            if word.lower() in word_to_coeff_mapping_p:
                weight = word_to_coeff_mapping_p[word.lower()]
            else:
                weight = word_to_coeff_mapping_n[word.lower()]

            if weight > 0:
                color = color_classes[pred]
            else:
                color = color_classes[opposite]
                weight *= -1
                weight *= 10

            highlighted_text.append('<span font-size:40px; ; style="background-color:rgba(' + color + ',' + str(
                weight) + ');">' + html_escape(word) + '</span>')

        else:
            highlighted_text.append(word)

    highlighted_text = ' '.join(highlighted_text)

    return highlighted_text


def display_LIME(data_original, data_clean, class_names=["Not Interviewed", "Interviwed"], save_to="lime.html"):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(data_clean, predict, num_features=60,
                                     num_samples=50, top_labels=2)
    l = exp.available_labels()
    run_info = exp.as_list(l[1])
    pred = l[1]

    # exp.show_in_notebook(text=False,predict_proba=True)
    # exp.save_to_file('test.html')
    # display(HTML(highlight_full_data(run_info,data_original,pred)))

    html_w = highlight_full_data(run_info, data_original, pred)
    with open(save_to, "w") as file:
        file.write(html_w)


# Example

def returnPrediction(data):
    # data = "drosophila a career in medicine was not the path i had originally envisioned for myself . as an undergraduate i dissected fruit fly brains and localized their neural pathways with fluorescence microscopy , constructed a novel behavior model using an iphone and a sheet of plexiglas , and discovered novel small molecule inhibitors with the potential to augment addiction behavior , all in pursuit of a career in basic science . i enjoyed the challenge of carefully sifting through mountains of genetic data and devising innovative experiments to harvest new information . i was fervently driven by the notion that my discoveries in the lab could ultimately contribute to the advancement of human health . by the end of my undergraduate experience , i recognized that medical research appealed to that part of me that recurrently asked and and yearned to probe deeper . “ how ? ” “ why ? ” however , there was something missing in my work , something that was revealed to me in a moment when volunteering in rural honduras . by pure chance , we became first responders to an accident ; a young child had sustained a traumatic leg injury while riding in a truck that rolled over at high speed . i held compression on her leg and helped her through the pain with soothing words of encouragement while the physicians tended to her injuries . though brief , the power of this interaction struck me at my core . i realized that i wanted to directly help people dealing with serious medical issues , not just from behind the lab bench , but with the knowledge and skills it takes to dramatically improve or even save their lives . i added pre med requirements to my curriculum , obtained clinical experience through volunteering , and pursued a medical degree . upon entering medical school i remained open to all areas of medicine , but early on i was captivated by the procedural specialties . in these fields the ailment could often be visualized , removed , reconstructed , or ablated . the ability to deliver a definitive therapy or diagnosis in a single session was intoxicating . to feed my desire for discovery while in medical school , i quickly acquired a research position within the vascular surgery department . i worked tirelessly in the vascular lab developing a novel model to characterize the potential clinical and biochemical impact of electronic cigarette vapor on abdominal aortic aneurysm formation , a model that is the driving force behind a national institute of health grant proposal . in vitro in vivo and with my sights initially set on vascular surgery i elected to do an interventional radiology rotation to broaden my understanding of endovascular procedures . there i learned that the breadth of interventional radiology is vast , and the diversity and complexity of procedures immense . we could repair ruptured aneurysms , embolize a traumatic bleed , and treat cancer , all with a few small incisions . moreover , from a technical standpoint i loved the continuous radiographic interpretation paired with the finesse of catheter manipulation . i felt no greater satisfaction than when devising novel interventional strategies for patients too ill to undergo an operation . i discovered that diagnostic and interventional radiology was my true passion . no specialty has greater opportunity for innovation and research than interventional radiology . i have recently undertaken numerous projects and have prepared several manuscripts on iliocaval reconstruction , interventional radiology operated endoscopy , and thoracic duct embolization and stenting . additionally , along with my mentors , i am in the process of patenting several novel devices including a pre shaped catheter for balloon occluded retrograde transvenous obliteration , a three dimensional snare device , a neuroprotective epidural balloon catheter , a low profile ablation balloon catheter , and an electromagnetic coupling catheter . the combination of my own curiosities , research interests , and interventional radiology experiences have aligned with my personal passions , making me not only a better physician radiologist , but also providing me"
    data_clean = text_cleaning(data)
    class_names = ["Not Interviewed", "Interviwed"]
    res=predict([data_clean])[0]
    prediction = class_names[np.argmax(res)]
    print("Prediction: ", prediction)

# Lime Example
    display_LIME(data,data_clean,save_to="templates/lime.html")

    return res
