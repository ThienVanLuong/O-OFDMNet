# basic import
import tensorflow as tf
import numpy as np
from keras import backend as K
from numpy import arange, newaxis 
from scipy.special import binom


def channel_test(U, noise_std):
    with tf.compat.v1.Session() as sess: 
        Y = U + K.random_normal(K.shape(U), mean=0, stddev=noise_std)
        Y = sess.run(Y)
    return Y


# generate data
def qam_constellation(M):
    a = 1/np.sqrt(2)
    b = np.sqrt(3)
    if M==4:
        QAM = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=complex) # gray mapping
    elif M==8:
        QAM = np.array([1, a+a*1j, -a+a*1j, 1j, a-a*1j, -1j, -1, -a-a*1j], dtype=complex) # 8PSK, not 8QAM indeed
        QAM = np.array([-3+1j, -3-1j, -1+1j, -1-1j, 3+1j, 3-1j, 1+1j, 1-1j], dtype=complex)  #gray qam

        real = np.array([0.624, -0.339, 1.020, -0.197, -0.962, 0.026, -0.603, 0.431])
        imag = np.array([0.946, 1.082, 0.065, -1.400, 0.344, 0.186, -0.538, -0.684])
        QAM = real + 1j*imag #Foschini

        QAM = np.array([0+0*1j, 1+0*1j, 0.2225+0.9749*1j, 0.6235+0.7818*1j, -0.2225-0.9749*1j, 0.6235-0.7818*1j,
                       -0.9010-0.4339*1j, -0.9010+0.4339*1j], dtype=complex) # 1-7
        QAM = np.array([1+b, 1-1j,-1-1j,(-1-b)*1j,1+1j,(1+b)*1j,-1-b,-1+1j], dtype=complex) #star QAM
    elif M==16:
        QAM = np.array([-3+3j, -3+1j, -3-3j, -3-1j,
                        -1+3j, -1+1j, -1-3j, -1-1j,
                        3+3j, 3+1j, 3-3j, 3-1j,
                        1+3j, 1+1j, 1-3j, 1-1j], dtype=complex)
    elif M==2:
        QAM = np.array([1, -1], dtype=complex) #BPSK
    else:
        raise ValueError('Modulation order must be in {2,4,8,16}.')
    return QAM
  

def generate_data_qam_symbols(M=4, N=16, num_samples=5, time_domain=False): 
    m = int(np.log2(M))
    QAM = qam_constellation(M)
    bits = np.random.binomial(n=1, p=0.5, size=(num_samples,N,m))
    sym_com = np.zeros((num_samples,N), dtype=complex)
    for i in range(num_samples):
        bit = bits[i]
        sym = np.zeros((N,), dtype=complex)
        for j in range(N):
            sym_id = bit[j].dot(2**np.arange(bit[j].size)[::-1])
            sym[j] = QAM[sym_id]
        if time_domain:
            sym = np.fft.ifft(sym)*np.sqrt(N)
        sym_com[i] = sym
        
    sym_r = np.real(sym_com) 
    sym_i = np.imag(sym_com)      
    sym_real = np.concatenate([sym_r,sym_i], axis=1)
    
    return bits, sym_com, sym_real

def qam_to_bits(X, M):   
    m = int(np.log2(M))
    bits = np.zeros((len(X),m), dtype=int)
    QAM = qam_constellation(M)
    bit_ref = arange(2**m)[:,newaxis] >> arange(m)[::-1] & 1
    for i in range(len(X)):
        x = X[i]
        for j in range(M):
            if x == QAM[j]:
                bits[i,:] = bit_ref[j,:]
    return bits

def prediction_to_bits(X_pred, M):
    N = int(X_pred.shape[1]/2)
    m = int(np.log2(M))
    bits_re = np.zeros((len(X_pred),N,m), dtype=int)
    if M==4:
        sym_real_re = np.sign(X_pred)   
        for j in range(len(X_pred)):
            sym_real_j = np.reshape(sym_real_re[j],(2,N))
            sym_com_j = sym_real_j[0,:] + 1j*sym_real_j[1,:]
            bits_re[j,:,:] = qam_to_bits(sym_com_j, M)
    else:
        QAM = qam_constellation(M) 
        for j in range(len(X_pred)):
            sym_real = np.reshape(X_pred[j],(2,N))
            sym_com = sym_real[0,:] + 1j*sym_real[1,:]
            sym_com_re = np.zeros(sym_com.shape, dtype=complex)
            for i in range(N):
                sym = sym_com[i]
                dis_m = np.abs(sym*np.ones((M,)) - QAM)
                sym_com_re[i] = QAM[np.argmin(dis_m)]
            bits_re[j,:,:] = qam_to_bits(sym_com_re, M)
    return bits_re

def evaluate_papr_o_ofdmnet(U):
    Za = np.abs(U)**2  
    papr = np.zeros((len(U),))
    for j in range(len(U)):
        papr[j] = np.max(Za[j])/np.mean(Za[j]) 
    return papr

def evaluate_papr_ofdm(X):
    papr_ofdm = np.zeros((len(X),))
    for i in range(len(X)):
        Z = X[i,:]
        N = len(Z)
        Zt = np.fft.ifft(Z)*np.sqrt(N)
        Za = np.abs(Zt)**2
        papr_ofdm[i] = np.max(Za)/np.mean(Za)
    return papr_ofdm

