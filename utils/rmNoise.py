from __future__ import division
import os
import numpy as np 
import wave,math
import utils.plot as plt
import utils.nextPow2 as np2

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Thres = 3
Expnt = 2.0
beta = 0.002
G = 0.9

def load_signal(filename):
    wf = wave.open(filename)
    params = wf.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    sound = wf.readframes(nframes) #-1 or nframes
    sound = np.fromstring(sound,dtype=np.int16)
    # sound = sound * 1.0/(max(abs(sound)))
    if sound.shape[0] == 2*nframes:
        sound = np.reshape(sound,[nframes,nchannels])
    else:
        sound = np.reshape(sound,[nframes-1,nchannels])
    sound = sound[:,0]
    wf.close()
    return sound,framerate

def get_lenParams(framerate,PERC = 50):
    '''
    compute params for future use
    '''
    len_ = 20 * framerate // 100
    len1 = len_ * PERC // 100
    len2 = len_ - len1
    return len_,len1,len2

def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 20:
            a = 1
    return a

def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 20:
            a = 1
    return a

def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list

def remove_noise(sound,framerate):
    '''
    remove noise from original sound
    '''
    len_,len1,len2 = get_lenParams(framerate)
    win = np.hamming(len_)
    winGain = len2 / sum(win)
    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * 2 ** (np2.next_pow2(len_))
    noise_mean = np.zeros(nFFT)

    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * sound[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # --- allocate memory and initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    nframes = len(sound) // len2 - 1
    xfinal = np.zeros(nframes * len2)

    for n in range(0,nframes):
        insign = win * sound[k-1:k+len_ - 1]
        spec = np.fft.fft(insign,nFFT)
        sig = abs(spec)
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
        
        if Expnt == 1.0:
            alpha = berouti1(SNRseg)
        else:
            alpha = berouti(SNRseg)

        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt 
        diffw = sub_speech - beta * noise_mu ** Expnt

        z = find_index(diffw)
        if len(z) > 0:
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            if SNRseg < Thres:
                noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  
                noise_mu = noise_temp ** (1 / Expnt) 

            sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
            x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
            xi = np.fft.ifft(x_phase).real
            xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
            x_old = xi[0 + len1:len_]
            k = k + len2
    return winGain,xfinal

def find_silenceBorder(sound,threshold=0.08):
    '''
    find sound value which is lower than threshold
    '''
    temp = sound * 1.0/(max(abs(sound)))
    ret = []
    for i in range(len(temp)):
        if abs(temp[i]) >= threshold:
            ret.append(i)
    return ret[0],ret[-1]

def truncate_silence(sound,trauncate_border = 12000):
    '''
    remove some noise in the begining
    '''
    left,right = find_silenceBorder(sound)
    if left < trauncate_border:
        left = trauncate_border
    for i in range(left):
        sound[i] = 0

    for i in range(right,len(sound)):
        sound[i] = 0
    return sound

def save_signal(filename,framerate,winGain,xfinal):
    '''
    save processed(remove noise) data
    '''
    nchannels = 1
    sampwidth = 2
    framerate = int(framerate)
    wave_data = (winGain * xfinal).astype(np.int16)
    # truncate_silence(wave_data)
    nframes = len(wave_data)
    comptype = "NONE"
    compname = "not compressed"
    wf = wave.open(filename,'wb')
    wf.setparams((nchannels, sampwidth, framerate, nframes,comptype, compname))
    wf.writeframes(wave_data.tostring())
    wf.close()
    
def load_originalData(dirname):
    '''
    load original data & remove noise
    '''
    if os.path.isdir(dirname):
        dirs = os.listdir(dirname)
        for directory in dirs:
            directoryName = os.path.join(dirname,directory)
            if os.path.isdir(directoryName):
                savedir = os.path.join(BASE_PATH,'data/processedData1',directory)
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                files = os.listdir(directoryName)
                for file in files:
                    if file.endswith('.wav'):
                        filename = os.path.join(directoryName,file)
                        print(filename)
                        try:
                            sound,framerate = load_signal(filename)
                            winGain,xfinal = remove_noise(sound,framerate)
                            savefile = os.path.join(savedir,file)
                            save_signal(savefile,framerate,winGain,xfinal)
                        except Exception as e:
                            print(e)

def load_appPreprocessedAudio(filename):
    '''
    the api of app, remove noise of the recorded audio
    '''
    sound,framerate = load_signal(filename)
    winGain,xfinal = remove_noise(sound,framerate)
    save_signal(filename,framerate,winGain,xfinal)
                        
if __name__ == '__main__':
    load_originalData(os.path.join(BASE_PATH,'data/originalData'))
