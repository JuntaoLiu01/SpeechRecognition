#coding:utf-8
from  __future__ import print_function
import os,pickle
import numpy as np 
from random import shuffle
from sklearn import svm 
from sklearn.externals import joblib

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH,'model/svm_20_75.pkl')
WORD_DICT = [
    '语音','余音','识别','失败','中国',
    '忠告','北京','背景','上海','商行',
    '复旦','饭店','Speech','Speaker','Signal',
    'File','Print','Open','Close','Project'
]

def pad_data(mfcc_feat):
    '''
    pad the matrix data to one dimension
    '''
    return mfcc_feat.reshape(mfcc_feat.size)

def load_data(dirname):
    '''
    load training datasets or test datasets
    '''
    X = [];y = []
    if os.path.isdir(dirname):
        files = os.listdir(dirname)
        shuffle(files)
        for file in files:
            if file.endswith('.npy'):
                filename = os.path.join(dirname,file)
                mfcc_feat = np.load(filename)
                mfcc_feat_pad = pad_data(mfcc_feat)
                X.append(mfcc_feat_pad)
                classType = int(file[0:2])
                y.append(classType)
    return X,y
    
def train(X,y):
    '''
    svm training,using SVC
    '''
    print('begin trainging....')
    clf = svm.SVC(decision_function_shape='ovr',gamma=0.00001,verbose=True)
    clf.fit(X,y)
    print('training done!')
    return clf

def predict(clf,X):
    '''
    predict 
    '''
    return clf.predict(X)

def eval_acc(pred,y):
    '''
    evaluate model's performance on training set or test set
    '''
    return np.sum(pred==y)/(1.0*len(y))

def load_model(train_dir,test_dir):
    '''
    load data sets & use svm to train data and predict
    '''
    X_train,y_train = load_data(train_dir)
    X_test,y_test = load_data(test_dir)

    clf = train(X_train,y_train)
    train_pred = predict(clf,X_train)
    test_pred = predict(clf,X_test)

    train_acc = eval_acc(train_pred,y_train)
    test_acc = eval_acc(test_pred,y_test)

    print("train accuracy: {:.2f}%".format(train_acc * 100))
    print("test accuracy: {:.2f}%".format(test_acc * 100))

    save_model(MODEL_PATH,clf)

def save_model(filename,clf):
    '''
    save training model on  local disk
    '''
    # with open(MODEL_PATH,'wb')  as fw:
    #     pickle.dump(clf,fw,protocol=2)
    joblib.dump(clf,filename,protocol=2)

def load_localModel(filename):
    '''
    load training model from local disk
    '''
    try:
        # print(filename)
        clf = joblib.load(filename)
        return clf
        # with open(MODEL_PATH,'rb') as fr:
        #     clf = pickle.load(fr)
            # return clf
    except Exception as e:
        print(e)
        print('model does not exist!')
        return None

def load_appRecogniseAudio(model,mfcc_feat):
    '''
    run app test,recognise audio
    '''
    if model:
        try:
            mfcc_feat_pad = pad_data(mfcc_feat)
            pred = model.predict([mfcc_feat_pad])
            print(WORD_DICT[pred[0]])
        except Exception as e:
            print(e)
    
# if __name__ == '__main__':
    # load_model(os.path.join(BASE_PATH,'data/trainData'),os.path.join(BASE_PATH,'data/testData'))