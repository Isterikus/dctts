#coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from hyperparameters import hp
from blocks import HighwayNetwork, Conv1d, Deconv1d, CharEmbed


class TextEncoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, dropout, hidden_units):
        super(TextEncoder, self).__init__()
        self.dropout_rate = dropout
        self.emb = CharEmbed(vocab_len, embedding_dim, 0)
        self.conv1 = Conv1d(in_channels=embedding_dim,
                            out_channels=2*hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='same',
                            dilation=1,
                            groups=1,
                            bias=True)

        #maybe add dropout layer instead of using F.dropout

        self.conv2 = Conv1d(in_channels=2*hidden_units,
                            out_channels=2*hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='same',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.highways1 = nn.ModuleList()
        for i in range(2):
            for j in range(4):
                self.highways1.append(
                HighwayNetwork(in_channels=2*hidden_units,
                               out_channels=2*hidden_units,
                               kernel=3,
                               stride=1,
                               padding='same',
                               dilation=3**j,
                               groups=1,
                               bias=True)
                )

        self.highways2 = nn.ModuleList()
        for i in range(2):
            self.highways2.append(
            HighwayNetwork(in_channels=2*hidden_units,
                           out_channels=2*hidden_units,
                           kernel=3,
                           stride=1,
                           padding='same',
                           dilation=1,
                           groups=1,
                           bias=True)
            )

        self.highways3 = nn.ModuleList()
        for i in range(2):
            self.highways3.append(
            HighwayNetwork(in_channels=2*hidden_units,
                           out_channels=2*hidden_units,
                           kernel=1,
                           stride=1,
                           padding='same',
                           dilation=1,
                           groups=1,
                           bias=True)
            )

    def forward(self, text_batch):
        x = self.emb(text_batch)        # [B, L, emb_dim]
        #x = x.transpose(1,2)            # [B, emb_dim, L]

        x = self.conv1(x)               # [B, 2*T2M_hid, L]
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) #set training param

        x = self.conv2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) #set Training param

        for highway in self.highways1:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways2:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways3:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        K, V = torch.chunk(x, 2, 1)

        return K, V

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, dropout, hidden_units):
        super(AudioEncoder, self).__init__()

        self.dropout_rate = dropout
        self.conv1 = Conv1d(in_channels=n_mels,
                            out_channels=hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='custom',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.conv2 = Conv1d(in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='custom',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.conv3 = Conv1d(in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='custom',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.highways1 = nn.ModuleList()
        for i in range(2):
            for j in range(4):
                self.highways1.append(
                HighwayNetwork(in_channels=hidden_units,
                               out_channels=hidden_units,
                               kernel=3,
                               stride=1,
                               padding='custom',
                               dilation=3**j,
                               groups=1,
                               bias=True)
                )

        self.highways2 = nn.ModuleList()
        for i in range(2):
            self.highways2.append(
            HighwayNetwork(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel=3,
                           stride=1,
                           padding='custom',
                           dilation=3,
                           groups=1,
                           bias=True)
            )

    def forward(self, mel_batch):
        #mel_batch shape: Batch x n_mels x T/r

        x = self.conv1(mel_batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) #set training param

        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training) #set Training param

        x = self.conv3(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways1:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways2:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x

def Attention(Q, K, V, prev_time=None, prev_max_attens=None):
    A = torch.bmm(K.transpose(1,2), Q) / np.sqrt(hp['T2M_hid'])

    A = F.softmax(A, 1)

    if not (prev_time is None or prev_max_attens is None):
        # forcibly incremental attention
        A[:, :, :prev_time+1].data.copy_(prev_max_attens[:, :, :prev_time+1].data)
        for i in range(int(A.size(0))):
            nt0 = torch.argmax(A[i, :, prev_time])
            nt1 = torch.argmax(A[i, :, prev_time + 1])

            if nt1 < nt0 - 1 or nt1 > nt0 + 3: #if nt1-nt0 is not in range [1, 3]
                nt0 = nt0 if nt0 + 1 < A.size(1) else nt0 - 1
                A[i, :, prev_time + 1].zero_()
                A[i, nt0 + 1, prev_time + 1] = 1
    R = torch.bmm(V, A)
    R = torch.cat((R, Q), 1)

    return R, A

class AudioDecoder(nn.Module):
    def __init__(self, n_mels, dropout, hidden_units):
        super(AudioDecoder, self).__init__()

        self.dropout_rate = dropout

        self.conv1 = Conv1d(in_channels=2*hidden_units,
                            out_channels=hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='custom',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.highways1 = nn.ModuleList()
        for i in range(4):
            self.highways1.append(
            HighwayNetwork(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel=3,
                           stride=1,
                           padding='custom',
                           dilation=3**i,
                           groups=1,
                           bias=True)
            )

        self.highways2 = nn.ModuleList()
        for i in range(2):
            self.highways2.append(
            HighwayNetwork(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel=3,
                           stride=1,
                           padding='custom',
                           dilation=1,
                           groups=1,
                           bias=True)
            )

        self.convs = nn.ModuleList()
        for i in range(3):
            self.convs.append(
            Conv1d(in_channels=hidden_units,
                   out_channels=hidden_units,
                   kernel_size=1,
                   stride=1,
                   padding='custom',
                   dilation=1,
                   groups=1,
                   bias=True)
            )

        self.logits = Conv1d(in_channels=hidden_units,
                             out_channels=n_mels,
                             kernel_size=1,
                             stride=1,
                             padding='custom',
                             dilation=1,
                             groups=1,
                             bias=True)

    def forward(self, R):

        x = self.conv1(R)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways1:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for highway in self.highways2:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)


        x = self.logits(x)
        #x = F.dropout(x, p=self.dropout_rate)

        mel_predicts = F.sigmoid(x)
        #x = logit(mel_predicts) since logit() is an inverse function to sigmoid

        return x, mel_predicts

class Text2Mel(nn.Module):
    def __init__(self, text_encoder, audio_encoder, audio_decoder, attention):
        super(Text2Mel, self).__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.attention = attention
        self.audio_decoder = audio_decoder

    def forward(self, text_batch, mel_batch):
        K, V = self.text_encoder(text_batch)
        Q = self.audio_encoder(mel_batch)
        R, A = self.attention(Q, K, V)
        logits, mel_predicts = self.audio_decoder(R)

        if self.training:
            return mel_predicts, logits, A
        else:
            return mel_predicts, A, K, V

    def infer(self, mel, K, V, prev_time, prev_max_attens):
        Q = self.audio_encoder(mel)
        R, A = self.attention(Q, K, V, prev_time, prev_max_attens)
        logits, mel_predicts = self.audio_decoder(R)

        return mel_predicts, A

class SSRN(nn.Module):
    def __init__(self, n_mels, n_fft, dropout, hidden_units):
        super(SSRN, self).__init__()

        self.dropout_rate = dropout

        self.conv1 = Conv1d(in_channels=n_mels,
                            out_channels=hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='same',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.highways1 = nn.ModuleList()
        for i in range(2):
            self.highways1.append(
            HighwayNetwork(in_channels=hidden_units,
                           out_channels=hidden_units,
                           kernel=3,
                           stride=1,
                           padding='same',
                           dilation=3**i,
                           groups=1,
                           bias=True)
                                  )
        self.transposed_convs_and_highways = nn.ModuleList()
        for i in range(2):
            self.transposed_convs_and_highways.append(
            Deconv1d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=2,
                     stride=2,
                     padding='same',
                     bias=True)
                                                     )
            for j in range(2):
                self.transposed_convs_and_highways.append(
                HighwayNetwork(in_channels=hidden_units,
                               out_channels=hidden_units,
                               kernel=3,
                               stride=1,
                               padding='same',
                               dilation=3**j,
                               groups=1,
                               bias=True)
                )

        self.conv2 = Conv1d(in_channels=hidden_units,
                            out_channels=2*hidden_units,
                            kernel_size=1,
                            stride=1,
                            padding='same',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.highways2 = nn.ModuleList()

        for i in range(2):
            self.highways2.append(
            HighwayNetwork(in_channels=2*hidden_units,
                           out_channels=2*hidden_units,
                           kernel=3,
                           stride=1,
                           padding='same',
                           dilation=1,
                           groups=1,
                           bias=True)
            )

        self.conv3 = Conv1d(in_channels=2*hidden_units,
                            out_channels=1+n_fft//2,
                            kernel_size=1,
                            stride=1,
                            padding='same',
                            dilation=1,
                            groups=1,
                            bias=True)

        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(
            Conv1d(in_channels=1+n_fft//2,
                   out_channels=1+n_fft//2,
                   kernel_size=1,
                   stride=1,
                   padding='same',
                   dilation=1,
                   groups=1,
                   bias=True)
            )

        self.logits = Conv1d(in_channels=1+n_fft//2,
                             out_channels=1+n_fft//2,
                             kernel_size=1,
                             stride=1,
                             padding='same',
                             dilation=1,
                             groups=1,
                             bias=True)

    def forward(self, mel_predicts):
        x = self.conv1(mel_predicts)
        x = F.dropout(x, p=self.dropout_rate)

        for highway in self.highways1:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate)

        for layer in self.transposed_convs_and_highways:
            x = layer(x)
            x = F.dropout(x, p=self.dropout_rate)

        x = self.conv2(x)

        for highway in self.highways2:
            x = highway(x)
            x = F.dropout(x, p=self.dropout_rate)

        x = self.conv3(x)
        x = F.dropout(x, p=self.dropout_rate)

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate)

        x = self.logits(x)
        #x = F.dropout(x, p=self.dropout_rate)

        linear_spectrum = F.sigmoid(x)

        return x, linear_spectrum


if __name__ == '__main__':
    #test class initializations
    text_encoder = TextEncoder(len(hp['vocab']), hp['emb'], hp['dropout'], hp['T2M_hid'])

    audio_encoder = AudioEncoder(hp['n_mels'], hp['dropout'], hp['T2M_hid'])

    audio_decoder = AudioDecoder(hp['n_mels'], hp['dropout'], hp['T2M_hid'])

    attention = Attention

    text2mel = Text2Mel(text_encoder, audio_encoder, audio_decoder, attention)

    ssrn = SSRN(hp['n_mels'], hp['n_fft'], hp['dropout'], hp['SSRN_hid'])

    print(text2mel)
    print(ssrn)
