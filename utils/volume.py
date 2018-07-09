import math 
import numpy as np 

def cal_volume(sound,frameSize,overLap):
    '''
    calculate abs sum of sound
    '''
    size = len(sound)
    step = frameSize - overLap
    frameNum = int(math.ceil(size * 1.0 / step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = sound[np.arange(i*step,min(i*step+frameSize,size))]
        curFrame = curFrame - np.median(curFrame) # zero-justified
        volume[i] = np.sum(np.abs(curFrame))
    return volume

def cal_volumeDB(sound,frameSize,overLap):
    '''
    calculate square sum of sound
    '''
    size = len(sound)
    step = frameSize - overLap
    frameNum = int(math.ceil(size * 1.0 / step))
    volume = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = sound[np.arange(i*step,min(i*step+frameSize,size))]
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        volume[i] = 10*np.log10(np.sum(curFrame*curFrame))
    return volume
