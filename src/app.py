#coding:utf-8
from __future__ import print_function
import os,sys,wave
import pyaudio
import numpy as np
import time

BASE_PATH  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_PATH)
import utils.rmNoise as rm
import utils.vad as vad
import utils.mfcc as mfcc
import utils.svm as svm

def record_audio(outfile,chunk=1024,format=pyaudio.paInt16,channels=2,seconds=2,rate=48000):
    '''
    record  audio for test
    '''
    p = pyaudio.PyAudio()
    stream = p.open(format=format,channels=channels,rate=rate,input=True,frames_per_buffer=chunk)
    print('*recording*')
    frames = []
    for i in range(0,int(rate/chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print('*done recording*')
    stream.stop_stream()
    stream.close()
    wf = wave.open(outfile, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_audio(filename,chunk=1024):
    '''
    play an audio from a wave file
    '''
    wf = wave.open(filename,'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=wf.getnchannels(),rate=wf.getframerate(),output=True)
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def app():
    filename = os.path.join(BASE_PATH,'data/app.wav')
    model = svm.load_localModel(svm.MODEL_PATH)
    while True:
        try:
            record_audio(filename)
            # play_audio(filename)
            # rm.load_appPreprocessedAudio(filename)
            sound,framerate = vad.load_appOriginalAudio(filename)
            mfcc_feat = mfcc.load_appProcessedAudio(sound,framerate)
            if mfcc_feat != []:
                svm.load_appRecogniseAudio(model,mfcc_feat)
            else:
                print("sorry!bad noise occurs,hard to deal")
            raw_input('print enter to continue:')
        except Exception as e:
            print(e)

def generate_testData(testfile):
    if os.path.isdir(testfile):
        files = os.listdir(testfile)
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(testfile,file)
                print(filename)
                savename = file[-9:-7] + file[-6:-4] + '.npy'
                # rm.load_appPreprocessedAudio(filename)
                sound,framerate = vad.load_appOriginalAudio(filename)
                mfcc_feat = mfcc.load_appProcessedAudio(sound,framerate)
                np.save(os.path.join(BASE_PATH,'test/20_75',savename),mfcc_feat)

def test_model(testfile):
    test_X,test_y = svm.load_data(testfile)
    model = svm.load_localModel(svm.MODEL_PATH)
    test_pred = model.predict(test_X)
    test_acc = svm.eval_acc(test_pred,test_y)
    print("test accuracy: {:.2f}%".format(test_acc * 100))

if __name__ == '__main__':
    app()
    # test_model(os.path.join(BASE_PATH,'test/20_75'))
    # generate_testData(os.path.join(BASE_PATH,'test/testData'))