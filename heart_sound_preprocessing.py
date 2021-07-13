import numpy as np
from scipy.signal import butter, lfilter, resample, hilbert, filtfilt
import scipy
from sklearn import preprocessing
import collections
import librosa
import pyhht
from pyhht.visualization import plot_imfs
from pyhht.emd import EMD
import pywt
from scipy import signal

# Normalization Function
def temporal_norm(input):
    x = preprocessing.minmax_scale(input)
    return x

def zero_temporal_norm(input):
    x = preprocessing.scale(input)
    return x

# Spike Remove Function
def spike_remove(sig, fs):
    '''
    The spike removal process works as follows:
    (1) The recording is divided into 500 ms windows.
    (2) The maximum absolute amplitude (MAA) in each window is found.
    (3) If at least one MAA exceeds three times the median value of the MAA's,
    the following steps were carried out. If not continue to point 4.
        (a) The window with the highest MAA was chosen.
        (b) In the chosen window, the location of the MAA point was identified as the top of the noise spike.
        (c) The beginning of the noise spike was defined as the last zero-crossing point before theMAA point.
        (d) The end of the spike was defined as the first zero-crossing point after the maximum point.
        (e) The defined noise spike was replaced by zeroes.
        (f) Resume at step 2.
    (4) Procedure completed.
    '''
    win_size = fs // 2
    dim = len(sig) // win_size
    sig_slice = np.reshape(sig[: win_size*dim], (dim, win_size))
    MAAs = np.max(abs(sig_slice), axis=1)
    while(len(MAAs[MAAs > np.median(MAAs)*4]) != 0):
        index = np.argmax(MAAs)
        max_window = sig_slice[index, :]
        max_index = np.argmax(max_window)
        ZCs = np.asarray(np.nonzero(librosa.zero_crossings(max_window))[0])
        if len(ZCs[ZCs < max_index]) > 0:
            spike_start = ZCs[ZCs < max_index][-1]
        else:
            spike_start = 0
        if len(ZCs[ZCs > max_index]) > 0:
            spike_end = ZCs[ZCs > max_index][0]+1
        else:
            spike_end = 500
        sig_slice[index, spike_start: spike_end] = 0.0001
        MAAs = np.max(abs(sig_slice), axis=1)

    despiked_sig = sig_slice.flatten()
    despiked_sig = np.append(despiked_sig, sig[len(despiked_sig): ])

    return despiked_sig

# homomorphic envelope with hilbert
def homomorphic_envelope(x, fs=1000, f_LPF=8, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    he[0] = he[1]
    return he

def hilbert_transform(x):
    """
    Computes modulus of the complex valued
    hilbert transform of x
    """
    return np.abs(hilbert(x))

# Bandpass filter function
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_ifft(X, Low_cutoff, High_cutoff, F_sample, M=None):
    """Bandpass filtering on a real signal using inverse FFT

    Inputs
    =======

    X: 1-D numpy array of floats, the real time domain signal (time series) to be filtered
    Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
    High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
    F_sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)

    Notes
    =====
    1. The input signal must be real, not imaginary nor complex
    2. The Filtered_signal will have only half of original amplitude. Use abs() to restore.
    3. In Numpy/Scipy, the frequencies goes from 0 to F_sample/2 and then from negative F_sample to 0.

    """

    import scipy, numpy
    if M == None: # if the number of points for FFT is not specified
        M = X.size # let M be the length of the time series
    Spectrum = scipy.fftpack.rfft(X, n=M)
    [Low_cutoff, High_cutoff, F_sample] = map(float, [Low_cutoff, High_cutoff, F_sample])

    #Convert cutoff frequencies into points on spectrum
    [Low_point, High_point] = map(lambda F: F/F_sample * M /2, [Low_cutoff, High_cutoff])# the division by 2 is because the spectrum is symmetric

    Filtered_spectrum = [Spectrum[i] if i >= Low_point and i <= High_point else 0.0 for i in range(M)] # Filtering
    Filtered_signal = scipy.fftpack.irfft(Filtered_spectrum, n=M)  # Construct filtered signal

    return Filtered_signal

def downsample(data, rate, new_rate):
    num = int(len(data) * new_rate / rate)
    y = scipy.signal.resample(data, num)
    return y

# Wiener filter function
def wiener(sig, mysize=None, noise=None):
    sig = np.asarray(sig)

    if mysize is None:
        mysize = [3] * len(sig)
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), sig.ndim)

    # Estimate the local mean
    lMean = scipy.signal.correlate(sig, np.ones(mysize), 'same') / np.product(mysize, axis=0)

    # Estimate the local variance
    lVar = (scipy.signal.correlate(sig ** 2, np.ones(mysize), 'same') / np.product(mysize, axis=0) - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = np.mean(np.ravel(lVar), axis=0)

    res = sig - lMean
    res *= (1 - noise / lVar)
    res += lMean
    out = np.where(lVar < noise, lMean, res)

    return out

def general_wiener(sig, freq, mysize, noise):
    N = len(sig)  # number of samples
    M = mysize  # length of Wiener filter
    Om0 = freq  # frequency of original signal
    N0 = noise  # PSD of additive white noise

    # estimate PSD
    f, Pss = signal.csd(sig, sig, nperseg=M)
    # compute Wiener filter
    G = np.fft.rfft(sig, M)
    H = 1/G * (np.abs(G)**2 / (np.abs(G)**2 + N0/Pss))
    H = H * np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2-8))  # shift for causal filter
    h = np.fft.irfft(H)
    # apply Wiener filter to observation
    y = np.convolve(x, h, mode='same')

    return y

def local_wiener(sig, mysize=21, local_size=10, fs=2000):
    sig = np.asarray(sig)
    print(sig.shape)

    pn = np.var(np.reshape(sig, (-1, local_size)), axis=-1)
    pn_med = np.quantile(pn, 0.25)
    pn_min = np.min(pn)
    pn_m = (pn_med + pn_min) / 2

    sig_filted = wiener(sig, mysize, noise=pn_m)

    return sig_filted

# HHT decomposer function
def hht_filter(sig, n_imfs_):
    decomposer = EMD(sig, n_imfs=n_imfs_)
    imfs = decomposer.decompose()

    return imfs[0: n_imfs_]

# Signal enhancement
def enhancement(hs, swin=30, lwin=300):
    #signal = remove_outliers(signal, 3)
    #hs = np.power(hs, 2)

    swin_filter = np.ones((swin, 1))
    lwin_filter = np.ones((lwin, 1))

    signal2_sf = signal.convolve(hs, swin_filter, 'same', 'auto')
    signal2_lf = signal.convolve(hs, lwin_filter, 'same', 'auto')

    coeff = signal2_sf / signal2_lf
    enhanced_signal = coeff * hs
    enhanced_signal = temporal_norm(enhanced_signal)

    return enhanced_signal

def wavelet_envelop(hs, wavelet, level_, keep_level_):
    # Wavelet Envelop
    coeffs = pywt.wavedec(hs, wavelet, level=level_)

    for i in range(level_+1):
        if i != keep_level_:
            coeffs[i] *= 0

    wavelet_feature = pywt.waverec(coeffs, wavelet)[: len(hs)]

    return wavelet_feature


def remove_outliers(hs, t):
    signalc = np.copy(hs)
    std = np.std(signalc)
    t_std = t * std
    outliers = np.where(np.abs(signalc) > t_std)
    signalc[outliers] = t_std
    #signalc = temporal_norm(signalc)

    return signalc
