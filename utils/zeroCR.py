import math
import numpy as np 

def zero_CR(sound,frameSize,overLap,epison=0.02):
    '''
    calculate zero crossing rate
    '''
    size = len(sound)
    step = frameSize - overLap
    frameNum = int(math.ceil(size*1.0/step))
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = sound[np.arange(i*step,min(i*step+frameSize,size))]
        curFrame = curFrame - np.mean(curFrame)-epison # zero-justified
        zcr[i] = sum(curFrame[0:-1] * curFrame[1::] < 0)
    return zcr