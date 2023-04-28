import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import pickle
import torch
import numpy as np
import itertools
import inspect
import copy
from hparams import hparams as hp
from eval_metrics import evaluate_metrics
from eval_metrics import evaluate_metrics_from_lists
from eval_metrics import combine_single_and_per_file_metrics
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from gensim.models.word2vec import Word2Vec
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence

import operator

import pdb 

def get_file_list(filepath, file_extension, recursive=True):
    '''
    @:param filepath: a string of directory
    @:param file_extension: a string of list of strings of the file extension wanted, format in, for example, '.xml', with the ".".
    @:return A list of all directories of files in given extension in given filepath.
    If recursive is True，search the directory recursively.
    '''
    pathlist = []
    if recursive:
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if type(file_extension) is list:
                    for exten in file_extension:
                        if file.endswith(exten):
                            pathlist.append(os.path.join(root, file))
                elif type(file_extension) is str:
                    if file.endswith(file_extension):
                        pathlist.append(os.path.join(root, file))
    else:
        files = os.listdir(filepath)
        for file in files:
            if type(file_extension) is list:
                for exten in file_extension:
                    if file.endswith(exten):
                        pathlist.append(os.path.join(filepath, file))
            elif type(file_extension) is str:
                if file.endswith(file_extension):
                    pathlist.append(os.path.join(filepath, file))
    if len(pathlist) == 0:
        print('Wrong or empty directory')
        raise FileNotFoundError
    return pathlist


def get_word_dict(word_dict_pickle_path, offset=0, reverse=False):
    word_dict_pickle = pickle.load(open(word_dict_pickle_path, 'rb'))
    word_dict = {}
    for i in range(0 + offset, len(word_dict_pickle) + offset):
        if reverse:
            word_dict[word_dict_pickle[i]] = i
        else:
            word_dict[i] = word_dict_pickle[i]
    return word_dict


def ind_to_str(sentence_ind, special_token, word_dict):
    sentence_str = []
    for s in sentence_ind:
        if word_dict[s] not in special_token:
            sentence_str.append(word_dict[s])
    return sentence_str

def gen_str(output_batch,word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    output_str = [' '.join(o) for o in output_str]
    return  output_str

def get_eval(output_batch, ref_batch, word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    ref_str = [[ind_to_str(r, special_token, word_dict) for r in ref] for ref in ref_batch]

    output_str = [' '.join(o) for o in output_str]
    ref_str = [[' '.join(r) for r in ref] for ref in ref_str]

    return  output_str, ref_str



def calculate_bleu(output, ref, word_dict_pickle_path, multi_ref=False):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = ind_to_str(output, special_token, word_dict)
    if multi_ref:
        ref_str = [ind_to_str(r, special_token, word_dict) for r in ref]
    else:
        ref_str = [ind_to_str(ref, special_token, word_dict)]

    gram_weights = []
    max_gram = 4
    for gram in range(1, max_gram + 1):
        weights = [0, 0, 0, 0]
        for i in range(gram):
            weights[i] = 1 / gram
        weights = tuple(weights)
        gram_weights.append(weights)

    score_list = []
    for weights in gram_weights:
        score = sentence_bleu(ref_str, output_str, weights=weights)
        score_list.append(score)
    return score_list, output_str, ref_str


def calculate_spider(output_batch, ref_batch, word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    ref_str = [[ind_to_str(r, special_token, word_dict) for r in ref] for ref in ref_batch]
    # pdb.set_trace()
    output_str = [' '.join(o) for o in output_str]
    ref_str = [[' '.join(r) for r in ref] for ref in ref_str]
    # print('calculate_spider is ',len(output_str))
    # print(output_str)
    # pdb.set_trace()
    # 
    metrics, per_file_metrics = evaluate_metrics_from_lists(output_str, ref_str)
    score = metrics['SPIDEr']

    return score, output_str, ref_str, metrics


def greedy_decode(model, src, max_len, start_symbol_ind=0):
    ### Transformer decoder
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    # memory = model.cnn(src)
    memory, memory_t, classifies, text_masks = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    for i in range(max_len - 1):
        # ys_i:(batch_size, T_pred=i+1)
        # pdb.set_trace()
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, memory_t, ys, target_mask=target_mask, text_masks=text_masks)  #  Transformer: (T_out, batch_size, nhid)  
        prob = model.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        # ys_i+1: (batch_size,T_pred=i+2)
    return ys, classifies

    ### Attdecoder
    # device = src.device
    # mem = model.encode(src)
    # output = model.translate_greedy(mem)
    # return  output
class Beam:
    """
    The beam class for handling beam search.
    partly adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/195f5ae17f572c22ff5229e52c2dd2254ad4e3db/onmt/translate/beam.py

    There are some place which needs improvement:
    1. The prev_beam should be separated as prev_beam and beam_score.
    The prev_beam should be a tensor and beam_score should be a numpy array,
    such that the beam advance() method could speeds up.
    2. Do not support advance function like length penalty.
    3. If the beam is done searching, it could quit from further computation.
    In here, an eos is simply appended and still go through the model in next iteration.
    """

    def __init__(self, beam_size, device, start_symbol_ind, end_symbol_ind):
        self.device = device
        self.beam_size = beam_size
        self.prev_beam = [[torch.ones(1).fill_(start_symbol_ind).long().to(device), 0]]
        self.start_symbol_ind = start_symbol_ind
        self.end_symbol_ind = end_symbol_ind
        self.eos_top = False
        self.finished = []
        self.first_time = True

    def advance(self, word_probs, first_time):  # word_probs: (beam_size, ntoken) or (1, ntoken) for the first time.

        if self.done():
            # if current beam is done, just add eos to the beam.
            for b in self.prev_beam:
                b[0] = torch.cat([b[0], torch.tensor(self.end_symbol_ind).unsqueeze(0).to(self.device)])
            return

        # in first time, the beam need not to align with each index.
        if first_time:  # word_probs:(1, ntoken)
            score, index = word_probs.squeeze(0).topk(self.beam_size, 0, True, True)  # get the initial topk
            self.prev_beam = []
            for s, ind in zip(score, index):
                # initialize each beam
                self.prev_beam.append([torch.tensor([self.start_symbol_ind, ind]).long().to(self.device), s.item()])
                self.prev_beam = self.sort_beam(self.prev_beam)
        else:  # word_probs:(beam_size, ntoken)
            score, index = word_probs.topk(self.beam_size, 1, True, True)  # get topk
            current_beam = [[b[0].clone().detach(), b[1]] for b in self.prev_beam for i in range(self.beam_size)]
            # repeat each beam beam_size times for global score comparison, need to detach each tensor copied.
            i = 0
            for score_beam, index_beam in zip(score, index):  # get topk scores and corresponding index for each beam
                for s, ind in zip(score_beam, index_beam):
                    current_beam[i][0] = torch.cat([current_beam[i][0], ind.unsqueeze(0)])
                    # append current index to beam
                    current_beam[i][1] += s.item()  # add the score
                    i += 1

            current_beam = self.sort_beam(current_beam)  # sort current beam
            if current_beam[0][0][-1] == self.end_symbol_ind:  # check if the top beam ends with eos
                self.eos_top = True

            # check for eos node and added them to finished beam list.
            # In the end, delete those nodes and do not let them have child note.
            delete_beam_index = []
            for i in range(len(current_beam)):
                if current_beam[i][0][-1] == self.end_symbol_ind:
                    delete_beam_index.append(i)
            for i in sorted(delete_beam_index, reverse=True):
                self.finished.append(current_beam[i])
                del current_beam[i]

            self.prev_beam = current_beam[:self.beam_size]  # get top beam_size beam
            # print(self.prev_beam)

    def done(self):
        # check if current beam is done searching
        return self.eos_top and len(self.finished) >= 1

    def get_current_state(self):
        # get current beams
        # print(self.prev_beam)
        return torch.stack([b[0] for b in self.prev_beam])

    def get_output(self):
        if len(self.finished) > 0:
            # sort the finished beam and return the sentence with the highest score.
            self.finished = self.sort_beam(self.finished)
            return self.finished[0][0]
        else:
            self.prev_beam = self.sort_beam(self.prev_beam)
            return self.prev_beam[0][0]

    def sort_beam(self, beam):
        # sort the beam according to the score
        return sorted(beam, key=lambda x: x[1], reverse=True)


def beam_search(model, src, max_len=30, start_symbol_ind=0, end_symbol_ind=9, beam_size=1):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    memory, _ = model.encode(src)  # memory:(T_mem,batch_size,nhid)
    # ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    first_time = True

    beam = [Beam(beam_size, device, start_symbol_ind, end_symbol_ind) for _ in range(batch_size)]  # a batch of beams

    for i in range(max_len):
        # end if all beams are done, or exceeds max length
        if all((b.done() for b in beam)):
            break

        # get current input
        ys = torch.cat([b.get_current_state() for b in beam], dim=0).to(device).requires_grad_(False)
        # pdb.set_trace()
        # get input mask
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, ntoken) for first time,
        # (T_out, batch_size*beam_size, ntoken) in other times
        out = F.log_softmax(out[-1, :], dim=-1)  # (batch_size, ntoken) for first time,
        # (batch_size*beam_size, ntoken) in other times

        beam_batch = 1 if first_time else beam_size
        # in the first run, a slice of 1 should be taken for each beam,
        # later, a slice of [beam_size] need to be taken for each beam.
        for j, b in enumerate(beam):
            b.advance(out[j * beam_batch:(j + 1) * beam_batch, :], first_time)  # update each beam

        if first_time:
            first_time = False  # reset the flag
            # after the first run, the beam expands, so the memory needs to expands too.
            memory = memory.repeat_interleave(beam_size, dim=1)

    output = [b.get_output() for b in beam]


    ### attdecoder

    return output

## beam search 
class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length, attend_a=None, attend_t=None):
        """
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.attend_a = attend_a
        self.attend_t = attend_t
    def __lt__(self, other):
        # return self.logp < other.logp
        # return -(self.logp) < -(other.logp)
        return (-self.eval()) < (-other.eval())
    def eval(self, alpha=0.92): # 0.9
        reward = 0
        # Add here a function for shaping a reward

        # return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        return (self.logp / float(self.leng) ** alpha) if self.leng else -1e6


def beam_decode(x, model, sos_ind=0, eos_ind=9, beam_width=5, top_k=1, kwp=False, max_len=30, visualization=False):###original
    """

    Args:
        x: input spectrogram (batch_size, time_frames, n_mels)
        model:
        sos_ind: index of '<sos>'
        eos_ind: index of '<eos>'
        beam_width: beam size
        top_k: how many sentences wanted to generate

    Returns:

    """
    decoded_batch = []

    device = x.device
    batch_size = x.shape[0]
    if kwp:
        encoded_features_audio, encoded_features_text, output, text_masks = model.encode(x)
    else:

        encoded_features, encoded_features_text, output, text_masks = model.encode(x)
    # audio features extracted by encoder, (time_frames, batch, nhid)
    if visualization:
        output_attend_a = []
        output_attend_t = []
    # decoding goes sentence by sentence
    for idx in range(batch_size):
        if kwp:
            # pdb.set_trace()
            encoded_feature_audio = encoded_features_audio[:, idx, :].unsqueeze(1)
            encoded_feature_text = encoded_features_text[:, idx, :].unsqueeze(1)
            # text_mask = text_masks[:, idx, :].unsqueeze(1)
        else:
            encoded_feature = encoded_features[:, idx, :].unsqueeze(1)
        
        # (time_frames, 1, n_hid)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_ind]]).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))
        # pdb.set_trace()
        # number_required = beam_width
        # number_required = 2
        # starting node -  previous node, word_id (sos_ind), logp, length
        if visualization:
            node = BeamSearchNode(None, decoder_input, 0, 1, [], [])
        else:
            node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            # pdb.set_trace()
            if n.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            if kwp:
                if visualization:
                    # all_attends_a = copy.deepcopy(n.attend_a)
                    # all_attends_t = copy.deepcopy(n.attend_t)
                    decoder_output, attn_weights, attn_weights_words = model.decode(encoded_feature_audio, encoded_feature_text, decoder_input)
                    attn_weights_ = copy.deepcopy(attn_weights)
                    attn_weights_words_ = copy.deepcopy(attn_weights_words)
                    # all_attends_a.append(attn_weights)
                    # all_attends_t.append(attn_weights_words)
                else:
                    decoder_output = model.decode(encoded_feature_audio, encoded_feature_text, decoder_input)
            else:
                decoder_output = model.decode(encoded_feature, None, decoder_input)
            log_prob = F.log_softmax(decoder_output[-1, :], dim=-1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(log_prob, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                if visualization:
                    node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1, attn_weights_, attn_weights_words_)
                else:
                    node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)

                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1
        # print('qsize:{}'.format(qsize))
        # choose n_best paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        # pdb.set_trace()
        output_attend_a_cur = []
        output_attend_t_cur = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordid[0, :])
            if visualization:
                output_attend_a_cur.append(n.attend_a)
                output_attend_t_cur.append(n.attend_t)
            # # back trace
            # while n.prevNode != None:
            #     n = n.prevNode
            #     utterance.append(n.wordid)
            #
            # utterance = utterance[::-1]
            # utterances.append(utterance)
        for i in range(top_k):
            decoded_batch.append(utterances[i])
            if visualization:
                output_attend_a.append(output_attend_a_cur[i])
                output_attend_t.append(output_attend_t_cur[i])
    if visualization:
        return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind), output, output_attend_a, output_attend_t
    else:
        return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)
###  ----------- ----------- ----------- ----------- ----------- ----------- -----------
'''
def beam_decode(x, model, sos_ind=0, eos_ind=9, beam_width=5, top_k=1, kwp=False, max_len=30):###original
    """

    Args:
        x: input spectrogram (batch_size, time_frames, n_mels)
        model:
        sos_ind: index of '<sos>'
        eos_ind: index of '<eos>'
        beam_width: beam size
        top_k: how many sentences wanted to generate

    Returns:

    """
    decoded_batch = []

    device = x.device
    batch_size = x.shape[0]
    if kwp:
        encoded_features_audio, encoded_features_text, output, text_masks = model.encode(x)
    else:

        encoded_features, encoded_features_text, output, text_masks = model.encode(x)
    # audio features extracted by encoder, (time_frames, batch, nhid)

    # decoding goes sentence by sentence
    for idx in range(batch_size):
        if kwp:
            # pdb.set_trace()
            encoded_feature_audio = encoded_features_audio[:, idx, :].unsqueeze(1)
            encoded_feature_text = encoded_features_text[:, idx, :].unsqueeze(1)
            # text_mask = text_masks[:, idx, :].unsqueeze(1)
        else:
            encoded_feature = encoded_features[:, idx, :].unsqueeze(1)
        
        # (time_frames, 1, n_hid)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_ind]]).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))
        # pdb.set_trace()
        # number_required = 4
        # starting node -  previous node, word_id (sos_ind), logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()
        temp_nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(), node))
    

        keeped_node_width = beam_width
        time_step = 0

        # start beam search
        while time_step < max_len:

            if len(endnodes) >= beam_width:
                break
            score, n = nodes.get()
            # pdb.set_trace()
            decoder_input = n.wordid
            if kwp:
                decoder_output = model.decode(encoded_feature_audio, encoded_feature_text, decoder_input)
            else:
                decoder_output = model.decode(encoded_feature, None, decoder_input)
            log_prob = F.log_softmax(decoder_output[-1, :], dim=-1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            
            log_prob, indexes = torch.topk(log_prob, beam_width)
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                score = -node.eval()
                temp_nodes.put((score, node))

            if nodes.qsize() == 0:
                for _ in range(beam_width):
                    score, node = temp_nodes.get()
                    if node.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                        endnodes.append((score, node))
                        keeped_node_width -= 1
                    else:
                        nodes.put((score, node))
                time_step += 1
                if time_step == max_len and keeped_node_width != 0:
                    for _ in range(keeped_node_width):
                        endnodes.append(nodes.get())
                temp_nodes = PriorityQueue()
            else:
                continue
        
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordid[0, :])
        for i in range(top_k):
            decoded_batch.append(utterances[i])
    
    return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)
'''
## ----------- ----------- ----------- ----------- ----------- ----------- ----------- -----------

# def beam_decode(x, model, sos_ind=0, eos_ind=9, beam_width=5, top_k=1, kwp=False, max_len=30):
#     """

#     Args:
#         x: input spectrogram (batch_size, time_frames, n_mels)
#         model:
#         sos_ind: index of '<sos>'
#         eos_ind: index of '<eos>'
#         beam_width: beam size
#         top_k: how many sentences wanted to generate

#     Returns:

#     """
#     decoded_batch = []

#     device = x.device
#     batch_size = x.shape[0]
#     if kwp:
#         encoded_features_audio, encoded_features_text, output, text_masks = model.encode(x)
#     else:

#         encoded_features, encoded_features_text, output, text_masks = model.encode(x)
#     # audio features extracted by encoder, (time_frames, batch, nhid)

#     # decoding goes sentence by sentence
#     for idx in range(batch_size):
#         if kwp:
#             # pdb.set_trace()
#             encoded_feature_audio = encoded_features_audio[:, idx, :].unsqueeze(1)
#             encoded_feature_text = encoded_features_text[:, idx, :].unsqueeze(1)
#             # text_mask = text_masks[:, idx, :].unsqueeze(1)
#         else:
#             encoded_feature = encoded_features[:, idx, :].unsqueeze(1)
        
#         # (time_frames, 1, n_hid)

#         # Start with the start of the sentence token
#         decoder_input = torch.LongTensor([[sos_ind]]).to(device)

#         # Number of sentence to generate
#         endnodes = []
#         # number_required = min((top_k + 1), top_k - len(endnodes))
#         number_required = beam_width

#         # starting node -  previous node, word_id (sos_ind), logp, length
#         node = BeamSearchNode(None, decoder_input, 0, 1)
#         nodes = PriorityQueue()
#         tmp_nodes = PriorityQueue()
#         # start the queue
#         nodes.put((-node.eval(), node))
        
#         qsize = 1
#         round = 0
#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             while True:
                
#                 if nodes.empty(): 
#                     break
#                 # fetch the best node
#                 score, n = nodes.get()
#                 decoder_input = n.wordid

#                 if (n.wordid[0, -1].item() == eos_ind and n.prevNode is not None) or (n.leng >= max_len -1):
#                     endnodes.append((score, n))
#                     # if we reached maximum # of sentences required
#                     if len(endnodes) >= number_required:
#                         break
#                     else:
#                         continue

#                 # decode for one step using decoder
#                 if kwp:
#                     decoder_output = model.decode(encoded_feature_audio, encoded_feature_text, decoder_input)
#                 else:
#                     decoder_output = model.decode(encoded_feature, None, decoder_input)
#                 log_prob = F.log_softmax(decoder_output[-1, :], dim=-1)

#                 # PUT HERE REAL BEAM SEARCH OF TOP
#                 log_prob, indexes = torch.topk(log_prob, beam_width)
#                 # nextnodes = []

#                 for new_k in range(beam_width):

#                     decoded_t = indexes[0][new_k].view(1, -1)
#                     log_p = log_prob[0][new_k].item()

#                     node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
#                     score = -node.eval()
#                     tmp_nodes.put((-node.eval(), node))
#                     # nextnodes.append((score, node))

#             if len(endnodes) >= number_required or tmp_nodes.empty(): 
#                 break
            
#             round += 1
#             assert nodes.empty()

#             # normally, tmp_nodes will have beam_width * beam_width candidates
#             # we only keep the most possible beam_width candidates
#             for i in range(beam_width):
#                 nodes.put(tmp_nodes.get())
#             tmp_nodes = PriorityQueue()
#             assert tmp_nodes.empty()


#                 # put them into queue
#                 # for i in range(len(nextnodes)):
#                 #     score, nn = nextnodes[i]
#                 #     nodes.put((score, nn))
#                 #     # increase qsize
#                 # qsize += len(nextnodes) - 1
#         # print('qsize:{}'.format(qsize))
#         # choose n_best paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(top_k)]

#         utterances = []
#         # pdb.set_trace()

#         for score, n in sorted(endnodes, key=operator.itemgetter(0)):
#             utterances.append(n.wordid[0, :])
#             # # back trace
#             # while n.prevNode != None:
#             #     n = n.prevNode
#             #     utterance.append(n.wordid)
#             #
#             # utterance = utterance[::-1]
#             # utterances.append(utterance)
#         for i in range(top_k):
#             decoded_batch.append(utterances[i])

#     return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)




def get_padding(tgt, tgt_len):
    # tgt: (batch_size, max_len)
    device = tgt.device
    batch_size = tgt.size()[0]
    max_len = tgt.size()[1]
    mask = torch.zeros(tgt.size()).type_as(tgt).to(device)
    for i in range(batch_size):
        d = tgt[i]
        num_pad = max_len-int(tgt_len[i].item())
        mask[i][max_len - num_pad:] = 1
        # tgt[i][max_len - num_pad:] = pad_idx

    # mask:(batch_size,max_len)
    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()
    return mask


def print_hparams(hp):
    attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
    return dict([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def find_item(data, key, query, item):
    """
    Search the query in key and take out the corresponding item.
    :param data:
    :param key:
    :param query:
    :param item:
    :return:
    """
    return data[data[key] == query][item].iloc[0]


# https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
# When smoothing=0.0, the output is almost the same as nn.CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=None, word_freq=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index
        # word_weight_data = pickle.load(open(word_freq,"rb"))
        # self.word_weight_data = torch.tensor(word_weight_data)
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        device = pred.device
        # self.word_weight_data = self.word_weight_data.to(device)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target.data == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        # return torch.mean(torch.sum(-true_dist * pred * self.word_weight_data, dim=self.dim))
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def align_word_embedding(word_dict_pickle_path, w2v_model_path, ntoken, nhid, load_type="word2vec"):
    word_dict = get_word_dict(word_dict_pickle_path)
    if load_type == "word2vec":
        model = Word2Vec.load(w2v_model_path)
        w2v_vocab = [k for k in model.wv.vocab.keys()]
    else:
        model = pickle.load(open(w2v_model_path,'rb'))
        w2v_vocab = [k for k in model.keys()]
    word_emb = torch.zeros((ntoken, nhid)).float()
    word_emb.uniform_(-0.1, 0.1)
    
    for i in range(len(word_dict)):
        word = word_dict[i]
        if word in w2v_vocab:
            if load_type == "word2vec":
                w2v_vector = model.wv[word]
            else :
                w2v_vector = model[word]
            word_emb[i] = torch.tensor(w2v_vector).float()
    if load_type == "bert":
        word_emb[-1] = torch.tensor(model['<pad>']).float()
    return word_emb


def tgt2onehot(target,class_num):
    target = target.unsqueeze_(2)
    batch_size = target.shape[0]
    t_steps = target.shape[1]
    y_one_hot = torch.zeros(batch_size,t_steps,class_num).scatter_(2,target,1)
    return y_one_hot

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    # print(mixup_lambda.dtype)
    device = x.device
    mixup_lambda = torch.tensor(mixup_lambda).to(device)
    # print(mixup_lambda.shape,type(mixup_lambda))
    out = (x[0 :: 2].transpose(0,-1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0,-1) * mixup_lambda[1 :: 2]).transpose(0,-1)
    # print("the ou dtype",out.dtype)
    return out
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


if __name__ == '__main__':
    import pdb 
    # pdb.set_trace()
    print('util')
    import torch
    mixup = Mixup(0.4)
    x = torch.rand(6,10)
    lambdas = mixup.get_lambda(6)
    print(x,lambdas)
    out = do_mixup(x,lambdas)
    print(out.shape,out.device)

