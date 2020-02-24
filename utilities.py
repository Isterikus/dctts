#coding: utf-8

import unicodedata
import re
import numpy as np
import librosa
from numba import jit
import torch
from scipy import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperparameters import hp

def load_vocabulary():
    chars_ids = {char: index for index, char in enumerate(hp['vocab'])}
    ids_chars = {index: char for index, char in enumerate(hp['vocab'])}

    return chars_ids, ids_chars

def norm_text(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                            if unicodedata.category(char) != 'Mn')

    text = text.lower()
    text = re.sub("[^{}]".format(hp['vocab']), " ", text)
    text = re.sub("[ ]+", " ", text)

    return text

def guide_attention(text_lengths, mel_lengths, r=None, c=None):
    b = len(text_lengths)
    if r is None:
        r = np.max(text_lengths)
    if c is None:
        c = np.max(mel_lengths)
    guide = np.ones((b, r, c), dtype=np.float32)
    mask = np.zeros((b, r, c), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(r):
            for t in range(c):
                if n < N and t < T:
                    W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (0.2 ** 2)))
                    M[n][t] = 1.0
                elif t >= T and n < N:
                    W[n][t] = 1.0 - np.exp(-((float(n - N - 1) / N)** 2 / (2.0 * (0.2 ** 2))))
    return guide, mask

@jit(nopython=True)
def guide_map(in_len, out_len, g=0.2):
    guide = np.ones((in_len, out_len), dtype=np.float32)
    for l in range(in_len): #amount of letters
        for t in range(out_len): #amount of mel timesteps
            guide[l][t] = 1.0 - np.exp(-(float(l) / in_len - float(t) / out_len) ** 2 / (2.0 * (g ** 2)))

    return guide

# tile torch tensor like in TF
def tile(a, dim, n_tile):
    '''
    Examples:

        t = torch.FloatTensor([[1,2,3],[4,5,6]])
        Out[54]:
        tensor([[ 1.,  2.,  3.],
                [ 4.,  5.,  6.]])
        Across dim 0:
        tile(t,0,3)
        Out[53]:
        tensor([[ 1.,  2.,  3.],
                [ 1.,  2.,  3.],
                [ 1.,  2.,  3.],
                [ 4.,  5.,  6.],
                [ 4.,  5.,  6.],
                [ 4.,  5.,  6.]])
        Across dim 1:
        tile(t,1,2)
        Out[55]:
        tensor([[ 1.,  1.,  2.,  2.,  3.,  3.],
                [ 4.,  4.,  5.,  5.,  6.,  6.]])
    '''
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def save_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr)

def save_attention(attention, path):
    fig, ax = plt.subplots()
    im = ax.imshow(attention)
    fig.colorbar(im)
    plt.savefig(path, format='png')
    plt.close(fig)

def save_spectrum(spectrum, path):
    fig, ax = plt.subplots()
    im = ax.imshow(np.flip(spectrum, 0), cmap="jet", aspect=0.2 * spectrum.shape[1] / spectrum.shape[0])
    fig.colorbar(im)
    plt.savefig(path, format='png')
    plt.close(fig)

def pad_1d(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)

def pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(0, 0), (b_pad, max_len - len(x[0]) - b_pad)],
               mode="constant", constant_values=0)
    return x

def spec2wav(spec):
    spec = np.clip(spec, 0, 1) * hp['max_db'] - hp['max_db'] + hp['ref_db']
    # to amplitude
    spec = np.power(10.0, spec * 0.05)

    # apply Griffin-Lim algorithm
    wav = GLA(spec**hp['power'])

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp['preemphasis']], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

# Griffin-Lim Algorithm
def GLA(spec):
    _mag = spec
    approximated_signal = librosa.istft(_mag, hp['hop_length'], win_length=hp['win_length'])
    for k in range(hp['n_iter']):
        estimation = librosa.stft(approximated_signal, hp['n_fft'], hp['hop_length'], win_length=hp['win_length'])
        angles = estimation / np.maximum(1e-8, np.abs(estimation)) #phases

        estimation = _mag * np.exp(1j * angles)
        approximated_signal = librosa.istft(estimation, hp['hop_length'], win_length=hp['win_length'])
    return approximated_signal

# Fast Griffin-Lim Algorithm
def FGLA(spec):
    _M = spec
    approximated_signal = librosa.istft(_M, hp['hop_length'], win_length=hp['win_length'])
    for k in range(hp['n_iter']):
        _D = librosa.stft(approximated_signal, hp['n_fft'], hp['hop_length'], win_length=hp['win_length'])
        _P = _D / np.maximum(1e-8, np.abs(_D))

        _D = _M * np.exp(1j * _P)
        _M = spectrogram + (.1 * np.abs(_D))
        approximated_signal = librosa.istft(_D, hp['hop_length'], win_length=hp['win_length'])
    return approximated_signal
