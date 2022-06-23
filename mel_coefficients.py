from scipy.signal import hamming
from scipy.fftpack import fft, fftshift, dct
import numpy as np
import matplotlib.pyplot as plt


def hertz_to_mel(freq):
    return 1125*np.log(1 + freq/700)
    
    
def mel_to_hertz(m):
    return 700*(np.exp(m/1125) - 1)


# calculate mel frequency filter bank
def mel_filterbank(nfft, filterbank, fs):
     
    # Clip at 8KHz
    lower_mel = hertz_to_mel(300)
    upper_mel = hertz_to_mel(8000)
    mel = np.linspace(lower_mel, upper_mel, filterbank+2)
    hertz = [mel_to_hertz(m) for m in mel]
    fbins = [int(hz * int(nfft/2+1)/fs) for hz in hertz]
    fbank = np.empty((int(nfft/2+1), filterbank))
    for i in range(1, filterbank+1):
        for k in range(int(nfft/2 + 1)):
            if k < fbins[i-1]:
                fbank[k, i-1] = 0
            elif k >= fbins[i-1] and k < fbins[i]:
                fbank[k, i-1] = (k - fbins[i-1])/(fbins[i] - fbins[i-1])
            elif k >= fbins[i] and k <= fbins[i+1]:
                fbank[k, i-1] = (fbins[i+1] - k)/(fbins[i+1] - fbins[i])
            else:
                fbank[k, i-1] = 0
     
# plotting mel freq filter banks, as shown in Q5, uncomment if you'd like to see
    #plt.figure(1)
    #xbins = fs*np.arange(0, nfft/2+1)/(nfft/2+1)
    #for i in range(filterbank):
        #plt.plot(xbins, fbank[:, i])
    #plt.axis(xmax=8000)
    #plt.xlabel('Frequency in Hz')
    #plt.ylabel('Amplitude')
    #plt.title('Mel Filterbank')
    #plt.show()
    
    return fbank
    

def mfcc(s, fs, filterbank):
  
    # segments of 25 ms with overlap of 10ms
    n_samples = np.int32(0.025*fs)
    overlap = np.int32(0.01*fs)
    n_frames = np.int32(np.ceil(len(s)/(n_samples-overlap)))
    # zero padding til len = n_frames
    padding = ((n_samples-overlap)*n_frames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((n_samples, n_frames))
    start = 0
    for i in range(n_frames):
        segment[:, i] = signal[start:start+n_samples]
        start = (n_samples-overlap)*i
    # compute periodogram
    nfft = 512
    periodogram = np.empty((n_frames, int(nfft/2 + 1)))
    for i in range(n_frames):
        x = segment[:, i] * hamming(n_samples)
        spectrum = fftshift(fft(x, nfft))
        periodogram[i, :] = abs(spectrum[int(nfft/2-1):])/n_samples
    # calculating mfccs
    fbank = mel_filterbank(nfft, filterbank, fs)
    # filterbank MFCCs for each frame
    mel_coeff = np.empty((filterbank, n_frames))
    for i in range(filterbank):
        for k in range(n_frames):
            mel_coeff[i, k] = np.sum(periodogram[k, :]*fbank[:, i])
    mel_coeff = np.log10(mel_coeff)
    mel_coeff = dct(mel_coeff)
    # exclude first coefficient (not useful)
    mel_coeff[0, :] = np.zeros(n_frames)
    return mel_coeff
