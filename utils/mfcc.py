from __future__ import print_function
import os
import numpy as np 

from splitSound import audio_toFrame
from splitSound import pre_emphasis
from splitSound import spectrum_power
from scipy.fftpack import dct

import scipy.io.wavfile as wav

try:
    xrange(1)
except:
    xrange=range

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
def calcMFCC_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''
    compute 13 MFCC,13 first-order differential coefficient,13 acceleration coefficient
    '''
    feat = calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy)
    result1 = derivate(feat)
    result2 = derivate(result1)
    result3 = np.concatenate((feat,result1),axis=1)
    result = np.concatenate((result3,result2),axis=1)
    return result

def derivate(feat,big_theta=2,cep_num=13):
    '''
    calculate the general transformation formula of the first-order coefficient or acceleration coefficient
    '''
    result = np.zeros(feat.shape)
    denominator = 0
    for theta in np.linspace(1,big_theta,big_theta):
        denominator = denominator + theta**2
    denominator = denominator*2
    for row in np.linspace(0,feat.shape[0]-1,feat.shape[0]):
        row  = int(row)
        tmp = np.zeros((cep_num,))
        numerator = np.zeros((cep_num,))
        for t in np.linspace(1,cep_num,cep_num):
            t = int(t)
            a=0;b=0;s=0
            for theta in np.linspace(1,big_theta,big_theta):
                theta = int(theta)
                if (t + theta) > cep_num:
                    a = 0
                else:
                    a = feat[row][t+theta-1]
                if (t-theta) < 1:
                    b = 0
                else:
                    b = feat[row][t-theta-1]
                s += theta*(a-b)
            numerator[t-1] = s
        tmp  = numerator * 1.0 / denominator
        result[row] = tmp
    return result


def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''
    compute 13 MFCC coefficient
    '''
    feat,energy = fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat = np.log(feat)
    feat = dct(feat,type=2,axis=1,norm='ortho')[:,:cep_num]
    feat = lifter(feat,cep_lifter)
    if appendEnergy:
        feat[:,0] = np.log(energy)
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''
    compute MFCC on signal,samplerate is sampling frequencty,
    win_length is the length of the window,win_step is the gap between windows,
    filters_num is the number of the Mel...
    '''
    high_freq = high_freq or samplerate/2
    signal = pre_emphasis(signal,pre_emphasis_coeff)
    frames = audio_toFrame(signal,win_length*samplerate,win_step*samplerate)
    spec_power = spectrum_power(frames,NFFT)
    energy = np.sum(spec_power,1)
    energy = np.where(energy == 0,np.finfo(float).eps,energy)
    fb = get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)
    feat = np.dot(spec_power,fb.T)
    feat = np.where(feat == 0,np.finfo(float).eps,feat)
    return feat,energy

def hz2mel(hz):
    '''
    change hz to mel hz
    '''
    return 2595 * np.log10(1+hz/700.0)

def mel2hz(mel):
    '''
    change mel hz to hz
    '''
    return 700 * (10**(mel/2595.0)-1)

def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''
    calculate the mel triangle filter
    '''
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = np.linspace(low_mel,high_mel,filters_num+2)
    hz_points = mel2hz(mel_points)
    bin = np.floor((NFFT+1) * hz_points/samplerate)
    fbank = np.zeros([filters_num,int(NFFT/2)+1])
    for j in xrange(filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''
    raise spectrum
    '''
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        return cepstra

def resize_mfcc(mfcc_feat,frames_num = 75):
    mfcc_frames = mfcc_feat.shape[0]
    # print(mfcc_frames)
    if mfcc_frames < frames_num:
        mfcc_feat = np.pad(mfcc_feat,((0,frames_num-mfcc_frames),(0,0)),mode='constant',constant_values=0)
    elif mfcc_frames > frames_num:
        mfcc_feat = mfcc_feat[0:frames_num,:]

    return mfcc_feat
    

def load_processedData(dirname):
    '''
    load processed signal data & compute mfcc feature
    '''
    # frames_num = np.zeros(100)
    if os.path.isdir(dirname):
        dirs = os.listdir(dirname)
        for directory in dirs:
            classType = str(directory)
            directoryName = os.path.join(dirname,directory)
            if os.path.isdir(directoryName):
                files = os.listdir(directoryName)
                for file in files:
                    if not file.endswith('.wav') : 
                        continue 
                    try:
                        print(os.path.join(directoryName,file))
                        (framerate,signal) = wav.read(os.path.join(directoryName,file))
                        mfcc_feat = calcMFCC_delta_delta(signal,framerate,win_length=0.05,win_step=0.02)
                        # frames_num[mfcc_feat.shape[0]] += 1
                        if mfcc_feat.shape[0] > 20 and mfcc_feat.shape[0] < 75:
                            mfcc_feat = resize_mfcc(mfcc_feat)
                            filename = classType + file[0:-4] + '.npy'
                            np.save(os.path.join(BASE_PATH,'data/mfccData/',classType,filename),mfcc_feat)
                    except Exception as e:
                        print(e)
    # for n,num in enumerate(frames_num):
    #     print(n+1,num)

def load_appProcessedAudio(signal,framerate):
    mfcc_feat = calcMFCC_delta_delta(signal,framerate,win_length=0.05,win_step=0.02) 
    if 20 < mfcc_feat.shape[0] < 75:
        mfcc_feat = resize_mfcc(mfcc_feat)
        return mfcc_feat
    else:
        return []
    
# if __name__ == '__main__':
#     dirname = os.path.join(BASE_PATH,'data/processedData/')
#     load_processedData(dirname)