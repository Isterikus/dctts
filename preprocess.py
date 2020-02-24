# coding: utf-8

import argparse
import librosa
import numpy as np
import os
from tqdm import tqdm

from hyperparameters import hp
from utilities import load_vocabulary, norm_text, guide_attention

def make_spectrograms(path):
    #input: path to wav file
    #output: mel -> [n_mels, T/r], mag -> [n_fft, T]

    y, samp_freq = librosa.load(path, sr=hp['sampling_frequency'])
    y, _ = librosa.effects.trim(y)     # remove silences
    y = np.append(y[0], y[1:] - hp['preemphasis']*y[:-1])

    #size: n_fft//2+1, times
    linear_spectrum = librosa.stft(y=y, n_fft = hp['n_fft'],
                                    hop_length=hp['hop_length'],
                                    win_length=hp['win_length'])

    magnitude_spectrum = np.abs(linear_spectrum)

    mel_basis = librosa.filters.mel(hp['sampling_frequency'], hp['n_fft'], hp['n_mels'])

    mel_spectrum = np.dot(mel_basis, magnitude_spectrum)

    #convert to decibels
    magnitude_spectrum[magnitude_spectrum < 1e-10] = 1e-10
    mel_spectrum[mel_spectrum < 1e-10] = 1e-10
    magnitude_spectrum = 20 * np.log10(magnitude_spectrum)
    mel_spectrum = 20 * np.log10(mel_spectrum)


    mel_spectrum = np.clip((mel_spectrum - hp['ref_db'] + hp['max_db']) / hp['max_db'], 1e-8, 1)
    magnitude_spectrum = np.clip((magnitude_spectrum - hp['ref_db'] + hp['max_db']) / hp['max_db'], 1e-8, 1)

    t = mel_spectrum.shape[1]
    num_paddings = hp['r'] - (t % hp['r']) if t % hp['r'] != 0 else 0

    magnitude_spectrum = np.pad(magnitude_spectrum, [[0, 0], [0,num_paddings]], mode='constant')
    mel_spectrum = np.pad(mel_spectrum, [[0, 0], [0,num_paddings]], mode='constant')

    mel_spectrum = mel_spectrum[:, ::hp['r']]

    return mel_spectrum, magnitude_spectrum

def get_texts(source):
    chars_ids, ids_chars = load_vocabulary()

    f_paths, text_lengths, texts = [], [], []
    meta = os.path.join(source, 'metadata.csv')
    with open(meta, encoding='utf-8') as f:
        for row in f:
            f_name, _, text = row.strip().split('|')
            f_path = os.path.join(source, 'wavs', f_name+'.wav')
            f_paths.append(f_path)
            text = norm_text(text)+'E'
            text = [chars_ids[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32))

    return f_paths, text_lengths, texts

def save_data(source, save_destination):
    print('Starting data preprocessing')
    os.makedirs(save_destination, exist_ok=True)
    train_data = open(os.path.join(save_destination, 'train_data.txt'), 'w', encoding='utf-8')
    mel_dest = os.path.join(save_destination, 'mels')
    mag_dest = os.path.join(save_destination, 'mags')
    text_dest = os.path.join(save_destination, 'texts')
    guide_dest = os.path.join(save_destination, 'guides')
    os.makedirs(mel_dest, exist_ok=True)
    os.makedirs(mag_dest, exist_ok=True)
    os.makedirs(text_dest, exist_ok=True)
    os.makedirs(guide_dest, exist_ok=True)
    print('Reading text files...')
    f_paths, text_lengths, texts = get_texts(source)
    print('Saving spectrograms...')
    for counter in tqdm(range(len(f_paths))):
        text = texts[counter]
        mel, mag = make_spectrograms(f_paths[counter])
        mel_file = 'mel_{:05d}.npy'.format(counter)
        mag_file = 'mag_{:05d}.npy'.format(counter)
        text_file = 'text_{:05d}.npy'.format(counter)
        text_length = text_lengths[counter]
        guide, mask = guide_attention([text_length], [mel.shape[-1]],
                                  200,
                                  240)
        guide = guide[0]
        guide_file = 'guide_{:05d}.npy'.format(counter)

        np.save(os.path.join(mel_dest, mel_file), mel.astype(np.float32))
        np.save(os.path.join(mag_dest, mag_file), mag.astype(np.float32))
        np.save(os.path.join(text_dest, text_file), text)
        np.save(os.path.join(guide_dest, guide_file), guide.astype(np.float32))

        train_data.write('|'.join([mel_file, mag_file, text_file, str(text_length), guide_file]) + '\n')
    train_data.close()

    print('Completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Folder with source data',
                         type=str)
    parser.add_argument('--destination', help='Folder in which to save preprocessed data',
                         type=str, default=hp['data_dir'])
    args = parser.parse_args()
    print('Source folder:\t', args.source)
    print('Destination folder:\t', args.destination)

    save_data(args.source, args.destination)
