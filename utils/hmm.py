from __future__ import division,print_function
import os,pickle
import numpy as np 
import warnings
from hmmlearn import hmm

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH,'model/gmmhmm')
WORD_DICT = [
    '语音','余音','识别','失败','中国',
    '中国','忠告','北京','背景','商行',
    '复旦','饭店','Speech','Speaker','Signal',
    'File','Print','Open','Close','Project'
]

TRANSMAT_PRIOR = np.array([[0.5,0.5,0. ],\
                           [0. ,0.5,0.5],\
                           [0. ,0. ,1. ]])
STARTPROB_PRIOR = np.array([[0.3,0.4,0.3]])
WEIGHTS_PRIOR = np.array([[0.6,0.4]])
class HmmModel():
    def __init__(self,Class):
        self.Class = Class
        self.model = hmm.GMMHMM(n_components = 3, n_mix = 2, \
                           transmat_prior = TRANSMAT_PRIOR, startprob_prior = STARTPROB_PRIOR, \
                           weights_prior = WEIGHTS_PRIOR,covariance_type = 'diag', n_iter = 50,params='mc')

def load_data(dirname):
    '''
    load mfcc data from local file
    '''
    features = [];labels = []
    if os.path.isdir(dirname):
        files = os.listdir(dirname)
        # shuffle(files)
        for file in files:
            if file.endswith('.npy'):
                filename = os.path.join(dirname,file)
                mfcc_feat = np.load(filename)
                features.append(mfcc_feat)
                classType = int(file[0:2])
                labels.append(classType)
    return features,labels

def split_data(features,labels):
    '''
    get the data for special class
    '''
    class_features = [None] * 20
    for j in range(len(class_features)):
        class_features[j] = np.zeros((0,39))
    for i in range(len(features)):
        for j in range(20):
            if labels[i] == j:
                class_features[j] = np.concatenate((class_features[j],features[i]))

    return class_features

def train(features,labels):
    '''
    gmmhmm training,using GMMHMM in hmm
    '''
    class_features = split_data(features,labels)
    hmmModels = []
    print('begin training...')
    for i in range(20):
        hmmModel = HmmModel(i)
        hmmModel.model.fit(class_features[i])
        hmmModels.append(hmmModel)
    print('training done!')
    return hmmModels

def predict(hmmModels,features):
    '''
    predict
    '''
    labels_pred = []
    for i in range(len(features)):
        scores = []
        for hmmModel in hmmModels:
            scores.append(hmmModel.model.score(features[i]))
        id = scores.index(max(scores))
        labels_pred.append(id)
    return labels_pred

def eval_acc(labels_pred,labels):
    '''
    evaluate model's performance on training set or test set
    '''
    return np.sum(labels_pred==labels)/(1.0*len(labels))

def save_model(hmmModels):
    '''
    save model on local disk
    '''
    for n,hmmModel in enumerate(hmmModels):
        model_name = os.path.join(MODEL_PATH,str(n)+'.pickle')
        with open(model_name,'wb') as fw:
            pickle.dump(hmmModel,fw,protocol=2)

def load_model(train_dir,test_dir):
    '''
    load data sets & use gmmhmm to train data and predict
    '''
    warnings.filterwarnings("ignore")
    train_features,train_labels = load_data(train_dir)
    test_features,test_labels = load_data(test_dir)

    hmmModels = train(train_features,train_labels)
    train_pred = predict(hmmModels,train_features)
    test_pred = predict(hmmModels,test_features)

    train_acc = eval_acc(train_pred,train_labels)
    test_acc = eval_acc(test_pred,test_labels)

    print("train accuracy: {:.2f}%".format(train_acc * 100))
    print("test accuracy: {:.2f}%".format(test_acc * 100))

    save_model(hmmModels)

if __name__ == '__main__':
    load_model(os.path.join(BASE_PATH,'data/trainData'),os.path.join(BASE_PATH,'data/testData'))









    