# coding: utf-8

import torch
import numpy as np
import argparse
import os
import codecs
from tqdm import tqdm

from utilities import spec2wav, load_vocabulary, norm_text, save_wav
from train import load_checkpoint, create_model

from hyperparameters import hp

cuda_present = torch.cuda.is_available()
if cuda_present:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def prepare_text_batch(text_file):
    lines = codecs.open(text_file, 'r', 'utf-8').readlines()
    char2idx, _ = load_vocabulary()
    text_batch = [norm_text(text) + 'E' for text in lines]
    max_text_batch_len = max([len(text) for text in text_batch])
    final_batch = []
    for text in text_batch:
        text = [char2idx[char] for char in text]
        text = np.concatenate((text, np.zeros(max_text_batch_len - len(text))))
        final_batch.append(text)
    return np.asarray(final_batch), max_text_batch_len


def infer(t2m_checkpoint, ssrn_checkpoint, text_file):
    t2m, _ = create_model('T2M')
    load_checkpoint(t2m_checkpoint, t2m, None, None, cuda_present)
    t2m = t2m.to(device)
    t2m.eval()

    #ssrn, _ = create_model('SSRN')
    #load_checkpoint(ssrn_checkpoint, ssrn, None, None, cuda_present)
    #ssrn = ssrn.to(device)
    #ssrn.eval()

    sample_dir = hp['sample_dir']
    os.makedirs(sample_dir, exist_ok=True)

    text_batch, max_text_batch_len = prepare_text_batch(text_file)
    num_texts = text_batch.shape[0]
    text_batch = torch.LongTensor(text_batch).to(device)

    max_mel_batch_len = max_text_batch_len + 50

    coarse_mels = torch.FloatTensor(np.zeros((len(text_batch), hp['n_mels'], max_mel_batch_len))).to(device)

    timesteps = max_mel_batch_len

    # initial step for t = 0, attentions = None
    new_coarse, prev_atten, K, V = t2m(text_batch, coarse_mels)
    coarse_mels[:, :, 1].data.copy_(new_coarse[:, :, 0].data)

    for t in tqdm(range(1, timesteps-1)):
        new_coarse, prev_atten = t2m.infer(coarse_mels, K, V, t-1, prev_atten)
        coarse_mels[:, :, t+1].data.copy_(new_coarse[:, :, t].data)

    np.save('test_mels3.npy', coarse_mels)

    #_, mags = ssrn(coarse_mels)

    '''for i in range(num_texts):
        attention_path = os.path.join(sample_dir, 'attention_{}.png'.format(i))
        spectrum_path = os.path.join(sample_dir, 'spectrum_{}.png'.format(i))
        wav_path = os.path.join(sample_dir, 'wav_{}.wav'.format(i))
        print('Running text {}'.format(i))

        save_attention(prev_atten[i].cpu().data, attention_path)
        save_spectrum(mags[i].cpu().data, spectrum_path)
        wav = spec2wav(mags[i].cpu().data.numpy())
        save_wav(wav, wav_path, sr=hp['sampling_frequency'])'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t2m_checkpoint', help='Path to Text2Mel network checkpoint',
                         type=str, default=None)
    parser.add_argument('--ssrn_checkpoint', help='Path to SSRN network checkpoint',
                         type=str, default=None)
    parser.add_argument('--text_file', help='Path to checkpoint to load from',
                         type=str, default=hp['txt_for_gen'])
    args = parser.parse_args()

    infer(args.t2m_checkpoint, args.ssrn_checkpoint, args.text_file)

    print('Done')
