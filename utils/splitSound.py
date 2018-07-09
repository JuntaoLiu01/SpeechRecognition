import numpy as np
import math

def audio_toFrame(sound,frame_length,frame_step,winfunc=lambda x:np.ones((x,))):
    '''
    change sound as frame
    '''
    size = len(sound)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    if size <= frame_length:
        frame_num = 1
    else:
        frame_num = 1 + int(math.ceil((1.0*size-frame_length)/frame_step))
    
    pad_length = int((frame_num-1)*frame_step+frame_length)
    zeros = np.zeros((pad_length-size,))
    pad_sound = np.concatenate((sound,zeros))
    indices = np.tile(np.arange(0,frame_length),(frame_num,1)) + np.tile(np.arange(0,frame_num*frame_step,frame_step),(frame_length,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = pad_sound[indices]
    win = np.tile(winfunc(frame_length),(frame_num,1))
    return frames*win

def def_frameSignal(frames,signal_length,frame_length,frame_step,winfunc=lambda x:np.ones((x,))):
    '''
    define frame signal,to reduce relation between frames
    '''

    signal_length = round(signal_length)
    frame_length = round(frame_length)
    frame_num = frames.shape[0]
    assert frame_length.shape[1] == frame_length, 'frame size is not correct,wrong column size '
    indices = np.tile(np.arange(0,frame_length),(frame_num,1))+np.tile(np.arange(0,frame_num*frame_step,frame_step),(frame_length,1)).T
    indices = np.array(indices,dtype=np.int32)
    pad_length = (frame_num-1)*frame_step + frame_length
    if signal_length <= 0:
        signal_length = pad_length
    recalc_signal = np.zeros((pad_length,))
    window_correction = np.zeros((pad_length,1))
    win = winfunc(frame_length)
    for i in range(0,frame_num):
        window_correction[indices[i,:]] = window_correction[indices[i,:]]+win+1e-15
        recalc_signal[indices[i,:]]=recalc_signal[indices[i,:]]+frames[i,:]
    recalc_signal = recalc_signal/window_correction
    return recalc_signal[0:signal_length]

def spectrum_magnitude(frames,NFFT):
    '''
    compute magnitude of each frame with FFT,frames:N*L --> N*NFFT
    '''
    complex_spectrum = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spectrum)

def spectrum_power(frames,NFFT):
    '''
    compute power spectral of each frames with FFT
    '''
    return 1.0/NFFT * np.square(spectrum_magnitude(frames,NFFT))

def log_spectrum_power(frames,NFFT,norm=1):
    '''
    compute the log format of power spectrum in each frame
    '''
    spec_power = spectrum_power(frames,NFFT)
    spec_power[spec_power<1e-30] = 1e-30
    log_spec_power = 10 * np.log10(spec_power)
    if norm:
        return log_spec_power - np.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(sound,coefficient=0.95):
    '''
    add previous emphsis on sound
    '''
    return np.append(sound[0],sound[1:]-coefficient*sound[:-1])

