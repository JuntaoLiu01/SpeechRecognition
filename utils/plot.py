import numpy as np 
import matplotlib.pyplot as plt 

def plot_originalSound(sound,framerate):
    '''
    plot original sound from file
    '''
    nframes = len(sound)
    time = np.arange(0,nframes) * (1.0/framerate)
    plt.plot(time,sound)
    plt.xlabel('Tims(s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_volume(sound,volume,framerate):
    '''
    plot volume of sound
    '''
    nframes = len(sound)
    frameNum = len(volume)
    time = np.arange(0,nframes) * (1.0/framerate)
    time1 = np.arange(0,frameNum) * (1.0 * nframes/frameNum) * (1.0/framerate)

    plt.subplot(211)
    plt.plot(time,sound) 
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.plot(time1, volume)
    plt.ylabel("Volume")
    # plt.subplot(313)
    # plt.plot(time1, volume2, c="g")  
    # plt.ylabel("Decibel(dB)") 
    plt.xlabel("Time(s)")
    plt.show()

def plot_zcr(sound,zcr,framerate):
    nframes = len(sound)
    frameNum = len(zcr)
    time = np.arange(0,nframes) * (1.0/framerate)
    time1 = np.arange(0,frameNum) * (1.0 * nframes/frameNum) * (1.0/framerate)

    plt.subplot(211)
    plt.plot(time,sound)
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.plot(time1,zcr)
    plt.ylabel('ZCR')
    plt.xlabel('Time(s)')
    plt.show()

def plot_volumeBorder(sound,volume,index,threshold,framerate):
    '''
    compare diffirent threshold of volume
    '''
    nframes = len(sound)
    frameNum = len(volume)
    step = 1.0 * nframes/frameNum
    timerate = 1.0/framerate
    index1 = index[0] * step * timerate;index2 = index[1] * step * timerate;index3 = index[2] * step * timerate
    threshold1 = threshold[0];threshold2 = threshold[1];threshold3 = threshold[2]

    end = nframes * timerate
    time = np.arange(0,nframes) * timerate
    time1 = np.arange(0,frameNum) * step * timerate

    plt.subplot(211)
    plt.plot(time,sound)
    plt.plot([index1,index1],[-1,1],'-r')
    plt.plot([index2,index2],[-1,1],'-g')
    plt.plot([index3,index3],[-1,1],'-b')
    plt.ylabel('Amplitude')

    plt.subplot(212)
    plt.plot(time1,volume)
    plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
    plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
    plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
    plt.legend()
    plt.ylabel('Volume')
    plt.xlabel('Time(s)')
    plt.show()

def plot_newBorder(sound,volume,index,framerate):
    '''
    plot new border of volume
    '''
    nframes = len(sound)
    frameNum = len(volume)
    step = 1.0 * nframes/frameNum
    timerate = 1.0/framerate
    index1 = index[0] * step * timerate;index2 = index[1] * step * timerate

    time = np.arange(0,nframes) * timerate

    plt.plot(time,sound)
    plt.plot([index1,index1],[-1,1],'-r')
    plt.plot([index2,index2],[-1,1],'-g')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    plt.show()

def plot_zcrBorder(sound,zcr,index,threshold,framerate):
    '''
    compare diffirent threshold of zcr
    '''
    nframes = len(sound)
    frameNum = len(zcr)
    step = 1.0 * nframes/frameNum
    timerate = 1.0/framerate
    index1 = index[0] * step * timerate;index2 = index[1] * step * timerate;index3 = index[2] * step * timerate
    threshold1 = threshold[0];threshold2 = threshold[1];threshold3 = threshold[2]

    end = nframes * timerate
    time = np.arange(0,nframes) * timerate
    time1 = np.arange(0,frameNum) * step * timerate

    plt.subplot(211)
    plt.plot(time,sound)
    plt.plot([index1,index1],[-1,1],'-r')
    plt.plot([index2,index2],[-1,1],'-g')
    plt.plot([index3,index3],[-1,1],'-b')
    plt.ylabel('Amplitude')

    plt.subplot(212)
    plt.plot(time1,zcr)
    plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
    plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
    plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
    plt.legend()
    plt.ylabel('ZCR')
    plt.xlabel('Time(s)')
    plt.show()

def plot_combinedBorder(sound,volume,zcr,index,framerate,name=None):
    nframes = len(sound)
    frameNum1 = len(volume)
    step1 = 1.0 * nframes/frameNum1
    frameNum2 = len(zcr)
    step2 = 1.0 * nframes/frameNum2
    timerate = 1.0/framerate
    index1 = index[0] * step1 * timerate;index2 = index[1] * step1 * timerate;index3 = index[2] * step2 * timerate
    
    time = np.arange(0,nframes) * timerate

    plt.plot(time,sound)
    plt.plot([index1,index1],[-1,1],'-r')
    plt.plot([index2,index2],[-1,1],'-g')
    plt.plot([index3,index3],[-1,1],'-b')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    if name:
        plt.savefig('/Users/juntaoliu/Desktop/debug3/'+name)
    else:
        plt.show()
    plt.close()


def plot_bestBorder(sound,index,framerate,name=None):
    index = 1.0 * index/framerate
    nframes = len(sound)
    time = np.arange(0,nframes) * (1.0/framerate)
    plt.plot(time,sound)
    plt.plot([index,index],[-1,1],'-r')
    plt.ylabel('Amplitude')
    plt.xlabel('Time(s)')
    if name:
        plt.savefig('/Users/juntaoliu/Desktop/debug2/'+name)
    else:
        plt.show()
    plt.close()