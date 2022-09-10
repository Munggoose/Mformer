import numpy as np

def Fft(input_data):
    n = len(input_data) 
    k = np.arange(n)
    Fs = 1/0.001
    T = n/Fs
    freq = k/T 
    freq = freq[range(int(n/2))]
    Y = np.fft.fft(input_data)/n 
    Y = Y[range(int(n/2))]
    sample = np.abs(Y)
    sample2 = Y
    return abs(Y),freq