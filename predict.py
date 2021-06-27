import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import shap

from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, remove_stopwords
import re
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords

import bertPredict as bp
nltk.download('wordnet')
shap.initjs()


def text_cleaning(data, steam=True, lemma=True, clean=True, min_lenght=2):
    words_sentences = []
    sentences = []

    for txt in data:
        orig_txt = txt
        txt = re.sub("none|other", "other", txt)
        filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]

        words = preprocessing.preprocess_string(txt, filters)
        stop_words = set(stopwords.words('english'))
        stop_words.remove("no")
        stop_words.remove("than")
        stop_words.remove("not")
        if clean:
            words = [w for w in words if
                     not w in stop_words and re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > min_lenght]
        else:
            words = [w for w in words if re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > 1]

        c_words = words

        if steam:
            porter = PorterStemmer()
            c_words = [porter.stem(word) for word in c_words]

        if lemma:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            c_words = [lem.lemmatize(word) for word in c_words]

        words_sentences.append(c_words)

        out = ""
        out = ' '.join(map(str, c_words))
        sentences.append(out)

    sentences = np.asarray(sentences)

    return sentences


# def parse1(CV):
#     CV = CV.replace('\r', '\n')
#     CV = CV.replace('\n\n', '\n')
#     CV = CV.replace('\n\n', '\n')
#     CV = CV.split('\n')
#     for j in range(len(CV)):
#         line = CV[j]
#         if line[:len('Medical Education')] == 'Medical Education':
#             medical_edu = ' '.join(CV[j + 1:j + 4])
#         if line[:len('Education')] == 'Education':
#             # edu = CV[j+1]
#             edu = ''
#             k = 0
#             while len(CV) > j + k + 1 and (CV[j + k + 1] != 'Membership and Honorary/Professional Societies' and CV[
#                 j + k + 1] != 'Medical School Awards' and CV[j + k + 1] != 'Volunteer Experience' and CV[
#                                                j + k + 1] != 'Certification/Licensure') and CV[
#                 j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
#                 edu += CV[j + k + 1]
#                 edu += '\n'
#                 k += 1
#         if line == 'Medical School Awards':
#             awards = ''
#             k = 0
#             while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Volunteer Experience' and CV[
#                 j + k + 1] != 'Average Hours/Week: ' and CV[j + k + 1] != 'Curriculum Vitae' and CV[
#                 j + k + 1] != 'Research Experience' and CV[j + k + 1] != 'Certification/Licensure' and CV[
#                 j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
#                 awards += CV[j + k + 1]
#                 awards += '\n'
#                 k += 1
#         if line == 'Certification/Licensure':
#             cert = CV[j + 1]
#
#         if line == 'Publications':
#             pub = ''
#             k = 0
#             while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Hobbies & Interests':
#                 pub += CV[j + k + 1]
#                 pub += '\n'
#                 k += 1
#
#     print(len(pub), len(awards), len(medical_edu), len(edu))
#     return pub, awards, medical_edu, edu
def parse(CV):
    pub = ''
    awards = ''
    medical_edu = ''
    edu = ''
    pub_count = 0

    CV = CV.replace('\r', '\n')
    CV = CV.replace('\n\n', '\n')
    CV = CV.replace('\n\n', '\n')
    CV = CV.split('\n')
    footer = False
    for j in range(len(CV)):
        line = CV[j]
        if line[:len('Medical Education')] == 'Medical Education':
            medical_edu = ' '.join(CV[j + 1:j + 4])
        if line[:len('Education')] == 'Education':
            # edu = CV[j+1]
            edu = ''
            k = 0
            while len(CV) > j + k + 1 and (CV[j + k + 1] != 'Membership and Honorary/Professional Societies' and CV[
                j + k + 1] != 'Medical School Awards' and CV[j + k + 1] != 'Volunteer Experience' and CV[
                                               j + k + 1] != 'Certification/Licensure') and CV[
                j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
                edu += CV[j + k + 1]
                edu += '\n'
                k += 1
        if line == 'Medical School Awards':
            awards = ''
            k = 0
            while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Volunteer Experience' and CV[
                j + k + 1] != 'Average Hours/Week: ' and CV[j + k + 1] != 'Curriculum Vitae' and CV[
                j + k + 1] != 'Research Experience' and CV[j + k + 1] != 'Certification/Licensure' and CV[
                j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
                awards += CV[j + k + 1]
                awards += '\n'
                k += 1
        if line == 'Certification/Licensure':
            cert = CV[j + 1]

        if line == 'Publications':
            pub = ''
            k = 0
            footer = False
            pub_count = 0
            while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Hobbies & Interests':
                if CV[j + k + 1][
                   :len(
                       'Emory University Program, Radiology-Diagnostic')] == 'Emory University Program, Radiology-Diagnostic':
                    footer = True
                if CV[j + k + 1][:len('Curriculum Vitae')] == 'Curriculum Vitae':
                    footer = False
                if footer == False:
                    pub += CV[j + k + 1]
                    pub += '\n'
                    if ('published' in CV[j + k + 1].lower()) or ('submitted' in CV[j + k + 1].lower()) or (
                            'presented at' in CV[j + k + 1].lower()) or ('presentation at' in CV[j + k + 1].lower()):
                        pub_count += 1
                k += 1

    return pub_count, pub, awards, medical_edu, edu


def predictPS(text):

    pred=bp.returnPrediction(text)
    print(type(pred))
    # ps_corpus_test = text_cleaning([text])
    # print(type(ps_corpus_test))
    # model = joblib.load("models/ps/model_logistic.pkl")
    # vectorizer = joblib.load("models/ps/modelvec_logistic.pkl")
    #
    # test_text = vectorizer.transform(ps_corpus_test)
    # pred = model.predict_proba(test_text)

    return pred[1]


def predictCV(text):
    pubcount, pub, awards, med_edu, edu = parse(text)

    pub_corpus_test = text_cleaning([pub])
    awd_corpus_test = text_cleaning([awards])
    edu_corpus_test = text_cleaning([edu])
    mededu_corpus_test = text_cleaning([med_edu])

    model_award = xgb.XGBClassifier()
    booster = xgb.Booster()
    award_vectorizer = joblib.load("models/awards/modelvec_xgbost.pkl")
    awd_test = award_vectorizer.transform(awd_corpus_test)
    booster.load_model("models/awards/model_xgbost.json")
    model_award._Booster = booster
    model_award._le = LabelEncoder().fit([0, 1])
    a = model_award.predict_proba(awd_test)
    print(a[:, 1])

    model_edu = xgb.XGBClassifier()
    booster = xgb.Booster()
    edu_vectorizer = joblib.load("models/education/modelvec_xgbost.pkl")
    edu_test = edu_vectorizer.transform(edu_corpus_test)
    booster.load_model("models/education/model_xgbost.json")
    model_edu._Booster = booster
    model_edu._le = LabelEncoder().fit([0, 1])
    e = model_edu.predict_proba(edu_test)
    print(e[:, 1])

    model_mededu = xgb.XGBClassifier()
    booster = xgb.Booster()
    mededu_vectorizer = joblib.load("models/med_education/modelvec_xgbost.pkl")
    mededu_test = mededu_vectorizer.transform(mededu_corpus_test)
    booster.load_model("models/med_education/model_xgbost.json")
    model_mededu._Booster = booster
    model_mededu._le = LabelEncoder().fit([0, 1])
    me = model_mededu.predict_proba(mededu_test)
    print(me[:, 1])

    model_pub = xgb.XGBClassifier()
    booster = xgb.Booster()
    pub_vectorizer = joblib.load("models/pub_vectorizer.pkl")
    pub_test = pub_vectorizer.transform(pub_corpus_test)
    booster.load_model("models/pub-classifier.json")
    model_pub._Booster = booster
    model_pub._le = LabelEncoder().fit([0, 1])
    p = model_pub.predict_proba(pub_test)
    print(p[:, 1])

    return [p[0][1], e[0][1], me[0][1], a[0][1]]


def getPrediction(input_data):
    final_model=joblib.load("models/final-classifier.pkl")
    # final_model = joblib.load("models/meta_learner/classifier.pkl")

    result = final_model.predict(input_data)

    explainer = shap.Explainer(final_model,feature_names=['ps', 'discrete', 'education', 'med_edu', 'awards'])
    shap_values = explainer(input_data)
    shap.plots.bar(shap_values, show=False)

    if result[0] == 0:
        plt.savefig('static/try0.png')
    else:
        plt.savefig('static/try1.png')
    print("Result: ", result[0])

    return result[0]
