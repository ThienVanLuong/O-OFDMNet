"""
Created by Thien Van Luong, Research Fellow at University of Southampton, UK.
He is now with the Faculty of CS, Phenikaa University, Vietnam.
Email: thienctnb@gmail.com

Paper O-OFDMNet: 
T. V. Luong, X. Zhang, L. Xiang, T. M. Hoang, C. Xu, P. Petropoulos, and L. Hanzo, 
"Deep learning-aided optical IM/DD OFDM approaches the throughput of RF-OFDM," 
IEEE J. Sel. Areas Commun., vol. 40, no. 1, pp. 212-226, Jan. 2022.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from keras import backend as K
from keras.layers import Lambda, Reshape, Dense, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

# basic import
from utils import *

# ----------------- parameters set up ------------
N = 64  # number of channel uses
M = 2   # total modulation size
m = int(np.log2(M))  # bits per symbol vector
SNRdB_train = 8
b_size = 1000
use_batch_norm = True
loss_scale = 0.001 #loss scaling factor

version = '01'
file_name = 'O_OFDMNet_N'+str(N)+'_M'+str(M)+'_ls'+str(loss_scale)+'_'+version

# network config, train, test data
norm_type = 0  # 0 per sample norm, 1 per batch norm
n_epoch = 100
l_rate = 0.001
hidden_layers_enc = [256]
hidden_layers_dec = [512, 256, 128]
train_size = 100000
test_size = 50000

load_model = True
retrain = False
loss_scale_recon = 1
loss_scale_papr = 0
new_version = version + 'a'

act_enc = 'relu'
act_dec = 'relu'
act_last_enc = 'sigmoid'
ini = 'glorot_uniform'

R = m
snr_train = 10 ** (SNRdB_train / 10.0)
noise_std = np.sqrt(1 / (2 * R * snr_train))

""" Basic functions """
def channel(U):
    Y = U + K.random_normal(K.shape(U), mean=0, stddev=noise_std)
    return Y

def ifft_transform(Z):
    Z = Reshape((N, 2))(Z)
    Zc = tf.complex(Z[:, :, 0], Z[:, :, 1])
    Zt = tf.signal.ifft(Zc) * np.sqrt(N)
    Zt_r = tf.math.real(Zt)
    Zt_i = tf.math.imag(Zt)
    Zt = K.concatenate([Zt_r, Zt_i])
    return Zt

def fft_transform(Z):
    Z = Reshape((N, 2))(Z)
    Zc = tf.complex(Z[:, :, 0], Z[:, :, 1])
    Zf = tf.signal.fft(Zc) / np.sqrt(N)
    Zf_r = tf.math.real(Zf)
    Zf_i = tf.math.imag(Zf)
    Zf = K.concatenate([Zf_r, Zf_i])
    return Zf

def get_papr(Zt):
    Za = K.abs(Zt) ** 2
    papr = K.max(Za) / K.mean(Za)
    return papr

""" build autoencoder model"""
# encoder
X = Input(shape=(2 * N,))

# output of encoder or transmitted vector
Zt = Lambda(lambda x: ifft_transform(x))(X)
enc = Zt
encoder_Zt = Model(X,Zt)

# encoder - time domain
if len(hidden_layers_enc) > 0:
    for i, num_nodes in enumerate(hidden_layers_enc):
        if use_batch_norm:
            enc = Dense(num_nodes, activation='linear', init=ini)(enc)
            enc = BatchNormalization(momentum=0.99, center=True, scale=True)(enc)
            enc = Activation(act_enc)(enc)
        else:
            enc = Dense(num_nodes, activation=act_enc, init=ini)(enc)
enc = Dense(N, activation=act_last_enc, init=ini)(enc)

if norm_type == 0:
    U = Lambda(lambda x: np.sqrt(N) * K.l2_normalize(x, axis=1))(enc)
else:
    U = Lambda(lambda x: x / tf.sqrt(tf.reduce_mean(tf.square(x))))(enc)

# Y is input of decoder, Y may include received signal y=hx+n and channel h
Y = Lambda(lambda x: channel(x))(U)
dec = Y

# decoder
for i, num_nodes in enumerate(hidden_layers_dec):
    if use_batch_norm:
        dec = Dense(num_nodes, activation='linear', init=ini)(dec)
        dec = BatchNormalization(momentum=0.99, center=True, scale=True)(dec)
        dec = Activation(act_dec)(dec)
    else:
        dec = Dense(num_nodes, activation=act_dec, init=ini)(dec)
Zt_hat = Dense(2 * N, activation='linear', init=ini)(dec)
X_hat = Lambda(lambda x: fft_transform(x))(Zt_hat)

AE = Model(X, X_hat)
#print(AE.summary())

# PAPR calculation
PAPR = Lambda(lambda x: get_papr(x))(Zt)
encoder_Zt = Model(X, Zt)
encoder_PAPR = Model(X, PAPR)

# model enc and dec
encoder = Model(X, U)
num_layers_decoder = len(AE.layers) - len(encoder.layers) - 1
X_enc = Input(shape=(N,))
deco = AE.layers[-num_layers_decoder](X_enc)  # first layer of decoder
for n in range(num_layers_decoder - 1):  # hidden and last layers of decoder
    deco = AE.layers[-num_layers_decoder + n + 1](deco)
    if n == num_layers_decoder - 3:
        decoder_Zt_hat = Model(X_enc,deco)
    if n == 1:
        decoder_1 = Model(X_enc,deco)
decoder = Model(X_enc, deco)

"""Training"""
# loss design
recon_loss = mse(X, X_hat)
total_loss = K.mean(recon_loss + loss_scale * PAPR)

# training
if not load_model:
    # generate train data
    _, _, sym_real = generate_data_qam_symbols(M=M, N=N, num_samples=train_size)
    AE.add_loss(total_loss)
    AE.compile(optimizer=Adam(lr=l_rate))
    AE.fit(sym_real, epochs=n_epoch, batch_size=b_size, verbose=2)

    encoder.save_weights('./models/encoder_'+file_name+'.h5')
    decoder.save_weights('./models/decoder_'+file_name+'.h5')
else:
    encoder.load_weights('./models/encoder_'+file_name+'.h5')
    decoder.load_weights('./models/decoder_'+file_name+'.h5')

if retrain:
    file_name = 'O_OFDMNet_N' + str(N) + '_M' + str(M) + '_ls' + str(loss_scale) + '_' + new_version
    total_loss = K.mean(loss_scale_recon*recon_loss + loss_scale_papr*PAPR)
    _, _, sym_real = generate_data_qam_symbols(M=M, N=N, num_samples=train_size)
    AE.add_loss(total_loss)
    AE.compile(optimizer=Adam(lr=l_rate))
    AE.fit(sym_real, epochs=n_epoch, batch_size=b_size, verbose=2)
    encoder.save_weights('./models/encoder_'+file_name+'.h5')
    decoder.save_weights('./models/decoder_'+file_name+'.h5')


"""Testing"""
bits, sym_com, sym_real = generate_data_qam_symbols(M=M, N=N, num_samples=test_size)

# BLER calculation and plot
SNRdB_test = list(np.linspace(0, 12, 7))
BER = [None] * len(SNRdB_test)

U = encoder.predict(sym_real)
power = 0
for i in range(len(U)):
    power = power + np.sum(U[i]*U[i])
print('Average power per N subcarriers ', power/len(U))

for n in range(0, len(SNRdB_test)):
    snr_test = 10 ** (SNRdB_test[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * snr_test))
    Y = channel_test(U, noise_std)
    X_pred = decoder.predict(Y)

    bits_re = prediction_to_bits(X_pred, M)
    bit_errors = (bits != bits_re).sum()
    BER[n] = bit_errors / test_size / m / N
    print('SNR:', SNRdB_test[n], 'BER:', BER[n])
    
for n in range(0, len(SNRdB_test)):
    print('SNR:', SNRdB_test[n], 'BER:', BER[n])

# PAPR evaluation and plot
papr_o_ofdmnet = evaluate_papr_o_ofdmnet(U)
papr_ofdm = evaluate_papr_ofdm(sym_com)

PAPR_dB = np.arange(12)
PAPR = 10 ** (PAPR_dB / 10.0)
cdf_papr_o_ofdmnet = [len(np.where(papr_o_ofdmnet > PAPR[i])[0]) / len(papr_o_ofdmnet) for i in range(len(PAPR))]
cdf_papr_ofdm = [len(np.where(papr_ofdm > PAPR[i])[0]) / len(papr_ofdm) for i in range(len(PAPR))]


plt.plot(SNRdB_test, BER, 'bo-', label='O-OFDMNet')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('Training SNR = ' + str(SNRdB_train) + 'dB, ' + 'loss scale = ' + str(loss_scale))
plt.grid()
plt.legend()
plt.savefig("./figs/BER_O-OFDMNet.png", dpi=100)
plt.show()

plt.plot(PAPR_dB, cdf_papr_o_ofdmnet, 'bo-', label='O-OFDMNet')
plt.plot(PAPR_dB, cdf_papr_ofdm, 'r+-', label='OFDM')
plt.yscale('log')
plt.xlabel('PAPR0 (dB)')
plt.ylabel('P(PAPR>PAPR0)')
plt.title('Training SNR = ' + str(SNRdB_train) + 'dB, ' + 'loss scale = ' + str(loss_scale))
plt.grid()
plt.legend()
plt.savefig("./figs/PAPR_O-OFDMNet.png", dpi=100)
plt.show()