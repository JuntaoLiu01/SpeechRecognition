from __future__ import print_function
from __future__ import division
import os,wave
import numpy as np
import struct
import volume as vp
import zeroCR as zeroCR
import plot as plt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nums = [0 for i in range(20)]

def load_signal(filename):
    '''
    load original signal
    '''
    wf = wave.open(filename,'r')
    params = wf.getparams()
    nchannels,sampwidth,framerate,nframes = params[:4]
    sound = wf.readframes(nframes) #-1 or nframes
    sound = np.fromstring(sound,dtype=np.int16)
    sound = sound * 1.0/(max(abs(sound)))
    if sound.shape[0] == 2*nframes:
        sound = np.reshape(sound,[nframes,nchannels])
    else:
        sound = np.reshape(sound,[nframes-1,nchannels])
    sound = sound[:,0]
    wf.close()

    return sound,framerate

def find_indexByVolume(volume,threshold):
    '''
    find border which split sound and others
    '''
    size = len(volume)
    index = []
    for i in range(size-1):
        if (volume[i] - threshold) * (volume[i+1] - threshold) < 0:
            index.append(i)
    return np.array([index[0],index[-1]])

def find_indexEnhanced(volume,index,epison=0.02,max_left_extended = 6,max_right_extended = 20):
    '''
    find better border index at the base of previous steps
    '''
    size = len(volume)
    i = index[0]
    while i > max(0,index[0]-max_left_extended):
        if volume[i] <= epison:
            break
        i =  i-1
    j = index[1]
    while j < min(size,index[1]+max_right_extended):
        if volume[j] <= epison:
            break
        j = j+1
    return np.array([i,j])

def find_indexByZCR(zcr,threshold):
    '''
    find sound and others border  by zcr
    '''
    size = len(zcr)
    index = []
    for i in range(size-1):
        if (zcr[i] - threshold) * (zcr[i+1] - threshold) < 0:
            index.append(i)
    return np.array([index[0],index[-1]])

def choose_border(sound,index,frames,framerate,left_margin=0.15,right_margin=0.2):
    '''
    choose best left border 
    '''
    step1 = 1.0 * frames[0]/frames[1]
    step2 = 1.0 * frames[0]/frames[2]
    timerate = 1.0/framerate
    volumeIndex =  index[0] * step1 * timerate
    newIndex = index[1] * step1 * timerate
    zcrIndex = index[2] * step2 * timerate

    if zcrIndex[0] < newIndex[0] and zcrIndex[0] >= volumeIndex[0]-left_margin:
        leftIndex = zcrIndex[0]
    else:
        leftIndex = newIndex[0]

    if zcrIndex[1] > newIndex[1] and zcrIndex[1] <= volumeIndex[1]+right_margin:
        rightIndex = zcrIndex[1]
    else:
        rightIndex = newIndex[1]
    return np.array([int(framerate * leftIndex),int(framerate * rightIndex)])
    
def save_signal(filepath,filename,sound,framerate):
    '''
    save processed signal data in local path
    '''
    outData = sound
    outfile = os.path.join(filepath,filename)
    outwave = wave.open(outfile, 'wb')

    nchannels = 1
    sampwidth = 2
    framerate = int(framerate)
    nframes = len(outData)
    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))

    for v in outData:
        value = int(v * 60000 / 2)
        # value = min(32768,value)
        # value = max(-32767,value)
        outwave.writeframes(struct.pack('h', value))
    outwave.close()

def test_volume(sound,framerate,frameSize,overLap):
    '''
    test function on valume
    '''
    # sound,framerate =  load_signal(filename)
    volume = vp.cal_volume(sound,frameSize,overLap)
    volume1 = vp.cal_volumeDB(sound,frameSize,overLap)
    plt.plot_volume(sound,volume,framerate)
    plt.plot_volume(sound,volume1,framerate)

def test_zcr(sound,framerate,frameSize,overLap):
    '''
    test function on zcr
    '''
    # sound,framerate =  load_signal(filename)
    zcr = zeroCR.zero_CR(sound,frameSize,overLap)
    plt.plot_zcr(sound,zcr,framerate)

def test_volumeThresh(sound,framerate,frameSize,overLap):
    '''
    test function on volume thresh
    '''
    # sound,framerate =  load_signal(filename)
    volume = vp.cal_volume(sound,frameSize,overLap)
    threshold1 = max(volume)*0.10
    threshold2 = min(volume)*10.0
    threshold3 = max(volume)*0.05+min(volume)*5.0
    index1 = find_indexByVolume(volume,threshold1) 
    index2 = find_indexByVolume(volume,threshold2) 
    index3 = find_indexByVolume(volume,threshold3)
    index = []
    index.append(index1);index.append(index2);index.append(index3)
    threshold = []
    threshold.append(threshold1);threshold.append(threshold2);threshold.append(threshold3)
    plt.plot_volumeBorder(sound,volume,index,threshold,framerate)

def test_newIndex(sound,framerate,frameSize,overLap):
    '''
    test function on newIndex
    '''
    # sound,framerate =  load_signal(filename)
    volume = vp.cal_volume(sound,frameSize,overLap)
    threshold3 = max(volume)*0.05+min(volume)*5.0
    index3 = find_indexByVolume(volume,threshold3)
    newIndex = find_indexEnhanced(volume,index3)
    index = []
    index.append(index3);index.append(newIndex)
    plt.plot_newBorder(sound,volume,index,framerate)

def test_zcrThresh(sound,framerate,frameSize,overLap):
    # sound,framerate =  load_signal(filename)
    zcr = zeroCR.zero_CR(sound,frameSize,overLap)
    zcrThresh1 = max(zcr)*0.10
    zcrThresh2 = min(zcr)*10.0 + 0.001
    zcrThresh3 = max(zcr)*0.05+min(zcr)*5.0
    zcrIndex1 = find_indexByZCR(zcr,zcrThresh1) 
    zcrIndex2 = find_indexByZCR(zcr,zcrThresh2) 
    zcrIndex3 = find_indexByZCR(zcr,zcrThresh3)
    zcrIndex = []
    zcrIndex.append(zcrIndex1);zcrIndex.append(zcrIndex2);zcrIndex.append(zcrIndex3)
    zcrThresh = []
    zcrThresh.append(zcrThresh1);zcrThresh.append(zcrThresh2);zcrThresh.append(zcrThresh3)
    plt.plot_zcrBorder(sound,zcr,zcrIndex,zcrThresh,framerate)

def test_combinedBorder(sound,framerate,frameSize,overLap1,overLap2,name=None):
    # sound,framerate =  load_signal(filename)
    volume = vp.cal_volume(sound,frameSize,overLap1)
    zcr = zeroCR.zero_CR(sound,frameSize,overLap2)
    threshold3 = max(volume)*0.05+min(volume)*5.0
    index3 = find_indexByVolume(volume,threshold3)
    
    newIndex = find_indexEnhanced(volume,index3)

    zcrThresh1 = max(zcr)*0.10
    zcrIndex1 = find_indexByZCR(zcr,zcrThresh1) 

    index = [index3,newIndex,zcrIndex1]
    plt.plot_combinedBorder(sound,volume,zcr,index,framerate,name)

def test_bestIndex(sound,framerate,frameSize,overLap1,overLap2,name=None):
    volume = vp.cal_volume(sound,frameSize,overLap1)
    zcr = zeroCR.zero_CR(sound,frameSize,overLap2)
    # plt.plot_orginalSound(sound,framerate)

    threshold3 = max(volume)*0.05+min(volume)*5.0
    index3 = find_indexByVolume(volume,threshold3)
    
    newIndex = find_indexEnhanced(volume,index3)

    zcrThresh1 = max(zcr)*0.10
    zcrIndex1 = find_indexByZCR(zcr,zcrThresh1) 

    index = [index3,newIndex,zcrIndex1]

    nframes = len(sound)
    frameNum1 = len(volume)
    frameNum2 = len(zcr)
    frames = [nframes,frameNum1,frameNum2]
    best_index = choose_border(sound,index,frames,framerate)
    # plt.plot_bestBorder(sound,best_index,framerate,name)
    return best_index

def truncate_silence(sound,trauncate_border=12000):
    sound[:trauncate_border] = 0
    return sound

def load_originalData(dirname):
    nums = [0 for i in range(20)]
    frameSize = 256;overLap1 = 128;overLap2 = 0  
    dirs = os.listdir(dirname)  
    for directory in dirs:
        directoryName = os.path.join(dirname,directory)
        if os.path.isdir(directoryName):
            files = os.listdir(directoryName)
            for file in files:
                if file.endswith('04.wav'):
                    filename = os.path.join(directoryName,file)
                    print(filename)
                    if os.path.isfile(filename):
                        try:
                            sound,framerate = load_signal(filename)
                            sound = truncate_silence(sound)
                            basename = os.path.basename(filename)
                            classType = basename[12:14]
                            filepath = os.path.join(BASE_PATH,'data/processedData2/',classType)
                            savename = str(nums[int(classType)]) + '.wav'
                            nums[int(classType)] = nums[int(classType)] + 1
                            best_index = test_bestIndex(sound,framerate,frameSize,overLap1,overLap2,classType+savename[:-4])
                            processedSound = sound[best_index[0]:best_index[1]]
                            save_signal(filepath,savename,processedSound,framerate)
                            test_combinedBorder(sound,framerate,frameSize,overLap1,overLap2,classType+savename[:-4])
                        except Exception as e:
                            print(e)

def load_appOriginalAudio(filename):
    frameSize=256;overLap1=128;overLap2=0
    try:
        sound,framerate = load_signal(filename)
        sound = truncate_silence(sound)
        # plt.plot_originalSound(sound,framerate)
        # test_combinedBorder(sound,framerate,frameSize,overLap1,overLap2)
    except:
        print('audio file does not exist! ')
        return None
    best_index = test_bestIndex(sound,framerate,frameSize,overLap1,overLap2)
    processedSound = sound[best_index[0]:best_index[1]]
    return processedSound,framerate


if __name__ == '__main__':
    dirname = os.path.join(BASE_PATH,'data/originalData')
    load_originalData(dirname)


