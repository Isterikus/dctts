# coding: utf-8

import torch
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from os import listdir
import numpy as np
import argparse

from hyperparameters import hp
from utilities import *
from networks import *

class LJSpeech(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.texts, self.mels, self.mags, self.guides = self.load_data()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        mel = self.mels[idx]
        mag = self.mags[idx]
        guide = self.guides[idx]

        return np.load(text), np.load(mel), np.load(mag), np.load(guide)

    def load_data(self):
        data = os.path.join(self.data_dir, 'train_data.txt')
        with open(data, encoding='utf-8') as f:
            rows = f.readlines()
        texts, mels, mags, guides = [], [], [], []
        for row in rows:
            mel, mag, text, _, guide, _ = row.strip().split('|')
            text_file = os.path.join(self.data_dir, 'texts', text)
            mel_file = os.path.join(self.data_dir, 'mels', mel)
            mag_file = os.path.join(self.data_dir, 'mags', mag)
            guide_file = os.path.join(self.data_dir, 'guides', guide)
            texts.append(text_file)
            mels.append(mel_file)
            mags.append(mag_file)
            guides.append(guide_file)

        return texts, mels, mags, guides

def custom_collate_fn(batch):

    input_lengths = [len(text[0]) for text in batch]
    max_text_len = max(input_lengths)

    mel_lengths = [mel[1].shape[1] for mel in batch]
    max_mel_len = max(mel_lengths)

    mag_lengths = [mag[2].shape[1] for mag in batch]
    max_mag_len = max(mag_lengths)

    text_batch = np.array([pad_1d(x[0], max_text_len) for x in batch], dtype=np.int32)

    mel_batch = np.array([pad_2d(x[1], max_mel_len) for x in batch], dtype=np.float32)

    mag_batch = np.array([pad_2d(x[2], max_mag_len) for x in batch], dtype=np.float32)

    guide_batch = np.array([x[3][:max_text_len,:max_mel_len] for x in batch], dtype=np.float32)

    return text_batch, mel_batch, mag_batch, guide_batch

def create_model(net):
    if net == 'T2M':
        print('Training Text to Mel network')

        text_encoder = TextEncoder(len(hp['vocab']), hp['emb'], hp['dropout'], hp['T2M_hid'])

        audio_encoder = AudioEncoder(hp['n_mels'], hp['dropout'], hp['T2M_hid'])

        audio_decoder = AudioDecoder(hp['n_mels'], hp['dropout'], hp['T2M_hid'])

        attention = Attention

        text2mel = Text2Mel(text_encoder, audio_encoder, audio_decoder, attention)

        train_func = train_T2M

        return text2mel, train_func

    elif net == 'SSRN':
        print('Training Spectrogram Super-resolution Network')

        ssrn = SSRN(hp['n_mels'], hp['n_fft'], hp['dropout'], hp['SSRN_hid'])

        train_func = train_SSRN

        return ssrn, train_func

def make_checkpoint(model, optimizer, checkpoint_dir, global_step, global_epoch):
    checkpoints = [ch for ch in listdir(checkpoint_dir) if ch.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    if len(checkpoints) > hp['checkpoints_to_keep']:
        os.remove(os.path.join(checkpoint_dir, checkpoints[0]))

    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step_{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": global_epoch,
    }, checkpoint_path)
    print("\nSaved checkpoint:", checkpoint_path)

def load_checkpoint(path, model, optimizer, reset_optimizer, cuda_present):
    global global_step, global_epoch

    print("Loading checkpoint from: {} ...".format(path))

    if cuda_present:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["state_dict"])
    if (optimizer is not None) and (reset_optimizer is not None): # optimizer is not needed for inference
        if not reset_optimizer:
            optimizer_state = checkpoint["optimizer"]
            if optimizer_state is not None:
                print("Loading optimizer state from {}".format(path))
                optimizer.load_state_dict(optimizer_state)
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    scale = hp['guide_weight'] * (hp['guide_decay'] ** global_step)

    return model

def save_output(outputs, attentions, writer, checkpoint_dir, global_step):
    #here only index's element is saved
    index = np.random.randint(0, len(outputs)) #choose it randomly

    if attentions is not None:
        mel = outputs[index].cpu().data.numpy()
        attention = attentions[index].cpu().data

        predicted_mel_dir = os.path.join(checkpoint_dir, 'predicted_mels')
        os.makedirs(predicted_mel_dir, exist_ok=True)
        mel_path = os.path.join(predicted_mel_dir, 'predict_mel_step_{:09d}.png'.format(global_step))

        attention_dir = os.path.join(checkpoint_dir, 'attentions')
        os.makedirs(attention_dir, exist_ok=True)
        attention_path = os.path.join(attention_dir, 'attention_step_{:09d}.png'.format(global_step))

        save_spectrum(mel, mel_path)
        save_attention(attention, attention_path)

    else:
        spectrum = outputs[index].cpu().data.numpy()

        predicted_wav_dir = os.path.join(checkpoint_dir, 'predicted_wavs')
        os.makedirs(predicted_wav_dir, exist_ok=True)
        wav_path = os.path.join(predicted_wav_dir, 'predict_step_{:09d}.wav'.format(global_step))

        spec_dir = os.path.join(checkpoint_dir, 'spectrums')
        os.makedirs(spec_dir, exist_ok=True)
        spec_path = os.path.join(spec_dir, 'spec_step_{:09d}.png'.format(global_step))

        wav = spec2wav(spectrum)

        #amplitude of sound should be in [-1, 1] -> divide output/np.max(np.abs(output))
        #writer.add_audio('Predicted wav', wav, global_step, sample_rate=hp['sample_rate'])

        save_wav(wav, wav_path, sr=hp['sampling_frequency'])
        save_spectrum(spectrum, spec_path)


def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
     # Noam scheme from tensor2tensor:
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


global_step = 0
global_epoch = 0
scale = hp['guide_weight']

def train_T2M(model, dataloader, optimizer, writer, checkpoint_dir, checkpoint_interval, num_epoches, device):
    model.train()
    global global_step, global_epoch, scale
    os.makedirs(checkpoint_dir, exist_ok=True)
    current_lr = hp['adam_learn_rate']

    criterion_output = nn.L1Loss().to(device)
    criterion_logits = nn.BCEWithLogitsLoss().to(device)
    criterion_attention = nn.L1Loss().to(device)

    while global_epoch < num_epoches:
        epoch_losses = 0.0

        for step, (text_batch, mel_batch, mag_batch, guide_batch) in tqdm(enumerate(dataloader)):
            #update learning rate according to given shedule
            if hp['use_lr_sheduling']:
                current_lr = noam_learning_rate_decay(hp['adam_learn_rate'], global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            #transform data to set it to device
            text_batch = torch.LongTensor(text_batch).to(device)

            coarse_batch = torch.FloatTensor(np.concatenate(
                            (np.zeros((mel_batch.shape[0], mel_batch.shape[1], 1)),
                             mel_batch[:, :, :-1]),
                            axis=2)).to(device)

            mel_batch = torch.FloatTensor(mel_batch).to(device)
            guide_batch = torch.FloatTensor(guide_batch).to(device)

            #run model
            output, logits, attention = model(text_batch, coarse_batch)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                print('\nMaking checkpoint and saving outputs at {} step'.format(global_step))
                make_checkpoint(model, optimizer, checkpoint_dir, global_step, global_epoch)
                save_output(output, attention, writer, checkpoint_dir, global_step)

            loss_output = criterion_output(output, mel_batch)
            loss_logits = criterion_logits(logits, mel_batch)

            attention_masks = torch.ones_like(attention)

            loss_atten = criterion_attention(
                guide_batch * attention * attention_masks,
                torch.zeros_like(attention)) * scale
            loss = loss_output + loss_logits + loss_atten

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # clip grad
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step() #update weights

            scale *= hp['guide_decay']
            if scale < hp['guide_lowbound']:
                scale = hp['guide_lowbound']

            #write logs
            writer.add_scalar('Losses/General_loss', float(loss.item()), global_step)
            writer.add_scalar('Losses/Output_loss', float(loss_output.item()), global_step)
            writer.add_scalar('Losses/Logit_loss', float(loss_logits.item()), global_step)
            writer.add_scalar('Losses/Attention_loss', float(loss_atten.item()), global_step)
            writer.add_scalar('Learning_rate', current_lr, global_step)

            epoch_losses += loss.item() # or =+ float(loss)
            global_step += 1

        average_epoch_loss = epoch_losses / len(dataloader)
        writer.add_scalar('Avarage loss per epoch', average_epoch_loss, global_epoch)

        print('Average loss on {} epoch: {}'.format(global_epoch, average_epoch_loss))
        global_epoch += 1

def train_SSRN(model, dataloader, optimizer, writer, checkpoint_dir, checkpoint_interval, num_epoches, device):
    model.train()
    global global_step, global_epoch
    os.makedirs(checkpoint_dir, exist_ok=True)
    current_lr = hp['adam_learn_rate']

    criterion_output = nn.L1Loss().to(device)
    criterion_logits = nn.BCEWithLogitsLoss().to(device)

    while global_epoch < num_epoches:
        epoch_losses = 0.0

        for step, (text_batch, mel_batch, mag_batch, guide_batch) in tqdm(enumerate(dataloader)):
            #update learning rate according to given shedule
            if hp['use_lr_sheduling']:
                current_lr = noam_learning_rate_decay(hp['adam_learn_rate'], global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            #optimizer.zero_grad()
            #transform data to set it to device

            mel_batch = torch.FloatTensor(mel_batch).to(device)
            mag_batch = torch.FloatTensor(mag_batch).to(device)

            #run model
            logits, output = model(mel_batch)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                print('\nMaking checkpoint and saving outputs at {} step'.format(global_step))
                make_checkpoint(model, optimizer, checkpoint_dir, global_step, global_epoch)
                save_output(output, None, writer, checkpoint_dir, global_step)

            loss_output = criterion_output(output, mag_batch)
            loss_logits = criterion_logits(logits, mag_batch)

            loss = loss_output + loss_logits

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # clip grad
            nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step() #update weights

            #write logs
            writer.add_scalar('Loss', float(loss.item()), global_step)
            writer.add_scalar('Output_loss', float(loss_output.item()), global_step)
            writer.add_scalar('Logit_loss', float(loss_logits.item()), global_step)
            writer.add_scalar('Learning_rate', current_lr, global_step)

            epoch_losses += loss.item()
            global_step += 1

        average_epoch_loss = epoch_losses / len(dataloader)
        writer.add_scalar('Avarage loss per epoch', average_epoch_loss, global_epoch)

        print('Average loss on {} epoch: {}'.format(global_epoch, average_epoch_loss))
        global_epoch += 1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Folder with preprocessed data',
                         type=str, default=hp['data_dir'])
    parser.add_argument('--net', help='Network to train',
                         type=str, default='T2M')
    parser.add_argument('--checkpoint_dir', help='Folder for model checkpoints',
                         type=str, default=hp['checkpoint_dir'])
    parser.add_argument('--log_dir', help='Folder for logs',
                         type=str, default=hp['log_dir'])
    parser.add_argument('--checkpoint_path', help='Path to checkpoint to load from',
                         type=str, default=None)
    parser.add_argument('--reset_optimizer', help='Whether to reset optimizer state',
                         type=bool, default=False)
    args = parser.parse_args()


    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    if args.net == 'T2M':
        checkpoint_dir = os.path.join(checkpoint_dir, 'T2M')
        log_dir = os.path.join(log_dir, 'T2M')
    elif args.net == 'SSRN':
        checkpoint_dir = os.path.join(checkpoint_dir, 'SSRN')
        log_dir = os.path.join(log_dir, 'SSRN')
    else:
        raise ValueError('Wrong network type (T2M or SSRN allowed)')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    lj_dataset = LJSpeech(args.data_dir)

    dataloader = DataLoader(lj_dataset, batch_size=hp['batch_size'],
                            shuffle=True, collate_fn=custom_collate_fn)

    model, train_func = create_model(args.net)

    cuda_present = torch.cuda.is_available()
    if cuda_present:
        torch.backends.cudnn.benchmark = False      #since input data has variable size
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)

    optimizer = optim.Adam(
    model.parameters(),
    lr = hp['adam_learn_rate'],
    betas = (hp['adam_beta1'], hp['adam_beta2']),
    eps = hp['adam_epsilon'],
    weight_decay = hp['weight_decay'],
    amsgrad = hp['amsgrad']
    )


    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, args.reset_optimizer, cuda_present)

    writer = SummaryWriter(log_dir=log_dir)

    try:
        train_func(
        model = model,
        dataloader = dataloader,
        optimizer = optimizer,
        writer = writer,
        checkpoint_dir = checkpoint_dir,
        checkpoint_interval=hp['checkpoint_interval'],
        num_epoches=hp['n_epoches'],
        device = device
        )

        print('Completed')
    except KeyboardInterrupt:
        make_checkpoint(model, optimizer, checkpoint_dir, global_step, global_epoch)
        print('Successfully created checkpoint. Exiting....')
