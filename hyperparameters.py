# coding: utf-8

hp = {
    #general
    'data_dir':'./Preprocessed_LJSpeech', #path to folder with preprocessed data
    'checkpoint_dir': './checkpoints/LJSpeech',
    'log_dir': './logdir/LJSpeech',
    'sample_dir': './samples',
    'txt_for_gen': './test_text.txt',

    #data preprocessing
    'sampling_frequency': 22050,
    'n_fft': 2048,
    'frame_shift': 0.0125,
    'frame_length': 0.05,
    'hop_length': 276,              #int(s_f*frame_shift)
    'win_length': 1102,             #int(s_f*frame_length)
    'n_mels': 80,
    'power': 1.5, #try 1.7
    'n_iter': 50, #set to 20
    'preemphasis': 0.97,
    'max_db': 100,
    'ref_db': 20,
    'r': 4,

    # networks parameters
    'dropout': 0.05,
    'emb': 128,
    'T2M_hid': 256,
    'SSRN_hid': 512,

    'checkpoint_interval': 1000,
    'checkpoints_to_keep': 15,

    'vocab': "PE abcdefghijklmnopqrstuvwxyz'.?",

    #training parameters
    'adam_learn_rate': 0.0002,
    'adam_beta1': 0.5,
    'adam_beta2': 0.9,
    'adam_epsilon': 1e-06,
    'weight_decay': 0.0,
    'amsgrad': False,
    'use_lr_sheduling': False,

    'guide_g': 0.2,  # bigger g, bigger guide area
    'guide_weight': 100.0,
    'guide_decay': 0.99999,
    'guide_lowbound': 1,

    'batch_size': 32, #16
    'n_epoches': 1200
}
