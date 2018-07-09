import os,sys,math

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import utils.rmNoise as rm
import utils.vad as vad 
import utils.mfcc as mfcc 
import utils.mv as mv
import utils.svm as svm

def noise_process(dirname):
    rm.load_originalData(dirname)

def vad_process(dirname):
    vad.load_originalData(dirname)

def mfcc_process(dirname):
    mfcc.load_processedData(dirname)

if __name__ == '__main__':
    # noise_process(os.path.join(BASE_PATH,'data/originalData/'))
    vad_process(os.path.join(BASE_PATH,'data/originalData/'))
    mfcc_process(os.path.join(BASE_PATH,'data/processedData2/'))
    mv.move_file(os.path.join(BASE_PATH,'data/mfccData/'))
    svm.load_model(os.path.join(BASE_PATH,'data/trainData'),os.path.join(BASE_PATH,'data/testData'))
