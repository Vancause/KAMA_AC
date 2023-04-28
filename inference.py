import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
# from data_handling import get_clotho_loader, get_test_data_loader
from model import AttModel, TransformerModel # , RNNModel, RNNModelSmall
import itertools
import numpy as np
import os
import sys
import logging
import csv
import random
from util import get_file_list, get_padding, print_hparams, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss, beam_search, \
        align_word_embedding, gen_str, Mixup, tgt2onehot, do_mixup, get_eval, get_word_dict, ind_to_str, beam_decode
from hparams import hparams
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn import metrics
import pdb
from scripts.cider import CiderScorer
import pickle
from warmup_scheduler import GradualWarmupScheduler
from librosa.feature import melspectrogram
import librosa
from typing import Union, List, Dict, Optional

def feature_extraction(audio_data: np.ndarray,
                       sr: int,
                       nb_fft: int,
                       hop_size: int,
                       nb_mels: int,
                       f_min: float,
                       f_max: float,
                       htk: bool,
                       power: float,
                       norm: bool,
                       window_function: str,
                       center: bool)\
        -> np.ndarray:
    """Feature extraction function.

    :param audio_data: Audio signal.
    :type audio_data: numpy.ndarray
    :param sr: Sampling frequency.
    :type sr: int
    :param nb_fft: Amount of FFT points.
    :type nb_fft: int
    :param hop_size: Hop size in samples.
    :type hop_size: int
    :param nb_mels: Amount of MEL bands.
    :type nb_mels: int
    :param f_min: Minimum frequency in Hertz for MEL band calculation.
    :type f_min: float
    :param f_max: Maximum frequency in Hertz for MEL band calculation.
    :type f_max: float|None
    :param htk: Use the HTK Toolbox formula instead of Auditory toolkit.
    :type htk: bool
    :param power: Power of the magnitude.
    :type power: float
    :param norm: Area normalization of MEL filters.
    :type norm: bool
    :param window_function: Window function.
    :type window_function: str
    :param center: Center the frame for FFT.
    :type center: bool
    :return: Log mel-bands energies of shape=(t, nb_mels)
    :rtype: numpy.ndarray
    """
    y = audio_data
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T
    logmel_spectrogram = librosa.core.power_to_db(
            mel_bands, ref=1.0, amin=1e-10, 
            top_db=None)
    logmel_spectrogram = logmel_spectrogram.astype(np.float32)        
    return logmel_spectrogram

def load_audio_file(audio_file: str, sr: int, mono: bool,
                    offset: Optional[float] = 0.0,
                    duration: Optional[Union[float, None]] = None)\
        -> np.ndarray:
    """Loads the data of an audio file.

    :param audio_file: The path of the audio file.
    :type audio_file: str
    :param sr: The sampling frequency to be used.
    :type sr: int
    :param mono: Turn to mono?
    :type mono: bool
    :param offset: Offset to be used (in seconds).
    :type offset: float
    :param duration: Duration of signal to load (in seconds).
    :type duration: float|None
    :return: The audio data.
    :rtype: numpy.ndarray
    """
    return librosa.load(path=audio_file, sr=sr, mono=mono,
                offset=offset, duration=duration)[0]


def generate(wav_file,beam_size=4):
    model.eval()
    
    audio_data = load_audio_file(wav_file, sr=sr, mono=mono)
    feature = feature_extraction(audio_data, sr=sr, nb_fft=nb_fft, hop_size=hop_size, \
                                 nb_mels=nb_mels, window_function=window_function, center=center, \
                                 f_min=f_min, f_max=f_max, htk=htk, power=power, norm=norm)
    feature = torch.tensor(feature).unsqueeze(0).to(device)
    output = model.beam_search(feature, beam_width=beam_size)
    
    out_sent =[] 
    for w_ind in output[0]:
        if w_ind == eos_ind:
            break 
        out_sent.append(word_dict[w_ind])
    out_sent = ' '.join(out_sent)
    # pdb.set_trace()
    
    return out_sent
if __name__ == '__main__':
    sos_ind = 0
    eos_ind = 9
    hp = hparams()
    mono = 'Yes'
    sr = 44100
    nb_fft = 1024
    hop_size = 512
    nb_mels = 64
    window_function = 'hann'
    center = 'Yes'
    f_min = .0
    f_max = None
    htk = 'No'
    power = 1.
    norm = 1

    wav_file_dir = 't_output/test.wav'
    checkpoint_dir = '256_lstm_tagloss_prev_tag/23.pt'
    device = torch.device(hp.device)
    pretrain_emb = None
    # loaa word_dict
    word_dict_pickle_path = './create_dataset/data/pickles/words_list.p'
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    model = AttModel(hp.ninp,hp.nhid,hp.output_dim_encoder,hp.emb_size,hp.dropout_p_encoder,
        hp.output_dim_h_decoder,hp.ntoken,hp.dropout_p_decoder,hp.max_out_t_steps,device,'tag',pretrain_emb,hp.tag_emb,
        hp.multiScale,hp.preword_emb,hp.two_stage_cnn,hp.usingLM, topk_keywords=hp.topk_keywords).to(device)
    checkpoint = torch.load(checkpoint_dir,map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    sentence = generate(wav_file=wav_file_dir)

    