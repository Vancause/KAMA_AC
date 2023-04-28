import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn import Module, GRU, Linear, Dropout
import random
# from torch.nn.modules.transformer import TransformerDecoder,TransformerDecoderLayer
from transformer_idea import TransformerDecoder,TransformerDecoderLayer
from queue import PriorityQueue
import operator
from hparams import hparams as hp
from encoder import Cnn10,init_layer,ResNet38,Tag
from util import do_mixup
import pickle
# from transformers import GPT2LMHeadModel, GPT2Config  # use gpt
import heapq as hq
from itertools import repeat
import pdb
# def myGPT():
#     device = torch.device(hp.device)
#     configuration = GPT2Config()
#     configuration.vocab_size = 4368
#     configuration.bos_token_id = 0
#     configuration.eos_token_id = 9
#     configuration.pad_token_id = 4037
#     model = GPT2LMHeadModel(configuration).to(device)
#     model.load_state_dict(torch.load('./models/finetune_gpt2_200.pt',map_location='cpu'))
#     return model

def sigmoid_decay(ite, epoch, rate=10.):
    ra = (float(epoch / 1.5) - float(ite))/float(epoch / rate)
    ra = 1.-(1. / (1.+ math.exp(ra)))
    return ra
def linear_decay(ite, epoch, rate=1.):
    ra = 1 - (1 - rate)*float(ite)/float(epoch)
    return ra

class EmbeddingC(nn.Module):
    def __init__(self, n_token, emb_dim, pretrain_emb=None):
        super(EmbeddingC, self).__init__()
        self.emb_dim = emb_dim
        self.n_token = n_token
        self.embedding = nn.Embedding(self.n_token, self.emb_dim)  # embedding layer
        if pretrain_emb is not None:
            self.embedding.weight.data = pretrain_emb

    def forward(self, x):
        out = self.embedding(x)
        return out


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, attention_dim, value_dim):
        super(Attention, self).__init__()
        self.Q = nn.Linear(query_dim, attention_dim)
        self.K = nn.Linear(key_dim, attention_dim)
        self.V = nn.Linear(key_dim, value_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, key, query, time_steps=None):
        if time_steps:
            query = query[:,:time_steps+1,:]
        att1 = self.K(key)
        att2 = self.Q(query)
        out = self.V(key)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class PrevWordAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(PrevWordAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, time_steps=None):
        x = encoder_out
        att1 = self.encoder_att(x)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (x * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class TaggingWordAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(TaggingWordAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden, time_steps=None):
        x = encoder_out
        att1 = self.encoder_att(x)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (x * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class Tag2Encodeout(nn.Module):
    def __init__(self, tag_dim, encoder_dim, attention_dim, project_k_v=0):
        self.query_linear = nn.Linear(tag_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.project_k_v = project_k_v
        if project_k_v:
            self.key_linear = nn.Linear(encoder_dim, attention_dim)
            self.value_linear = nn.Linear(encoder_dim, attention_dim)

    
    def forward(self, query, key, value):
        q = self.query_linear(query)
        if self.project_k_v:
            key = self.key_linear(key)
            value = self.value_linear(value)
        dot = torch.bmm(query,key.transpose(-1,-2))
        score = self.softmax(dot)
        att_res = torch.bmm(score,value).mean(dim=1)
        return att_res, score
 
class AttDecoder(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 att_dim: int,
                 emb_dim: int,
                 maxlength: int,
                 nb_classes: int,
                 dropout_p: float,
                 device,
                 pretrain_emb=None,
                 tag_emb=False,
                 multiScale=False,
                 preword_emb=False,
                 topk_keywords=5,
                 dataset='Clotho') \
            -> None:
        """Decoder with attention.
        :param input_dim: Input features in the decoder.
        :type input_dim: int
        :param output_dim: Output features of the RNN.
        :type output_dim: int
        :param nb_classes: Number of output classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        """
        super().__init__()

        if multiScale:
            self.attention = Attention(input_dim, output_dim, att_dim, output_dim)  # h is query, its dim :512
            self.attention1 = Attention(input_dim, output_dim, att_dim, output_dim)
            self.attention2 = Attention(input_dim, output_dim, att_dim, output_dim)

            self.init_h = nn.Linear(input_dim * 3, output_dim)
            self.init_c = nn.Linear(input_dim * 3, output_dim)
            self.linear_feature = nn.Linear(2048, input_dim)
            self.linear_feature1 = nn.Linear(128, input_dim)
            self.linear_feature2 = nn.Linear(256, input_dim)
        else:
            self.attention = Attention(input_dim, output_dim, att_dim, output_dim)
            self.init_h = nn.Linear(input_dim, output_dim)
            self.init_c = nn.Linear(input_dim, output_dim)
            self.linear_feature = nn.Linear(2048, input_dim)

        if tag_emb:
            if dataset == 'Clotho':
                self.tagging_to_embs = torch.tensor(pickle.load(open(hp.tagging_to_embs, 'rb')))
            else:
                self.tagging_to_embs = torch.tensor(pickle.load(open(hp.tagging_to_embs_audiocaps, 'rb')))
            self.taggingword_attention = Attention(emb_dim, output_dim, att_dim, output_dim)
            self.f_beta_taggingword = nn.Linear(input_dim, output_dim)
        if preword_emb:
            self.preword_attention = Attention(emb_dim, output_dim, att_dim, output_dim)
            self.f_beta_preword = nn.Linear(input_dim, output_dim)  # linear layer to create a sigmoid-activated gate

        self.tag_emb = tag_emb
        self.multiScale = multiScale
        self.preword_emb = preword_emb
        self.nb_classes = nb_classes
        self.dropout: Module = Dropout(p=dropout_p)
        self.decode_step = nn.LSTMCell(input_dim, output_dim, bias=True)
        self.f_beta = nn.Linear(input_dim, output_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.maxlength = maxlength
        self.classifier: Module = Linear(
            in_features=output_dim,
            out_features=nb_classes)
        self.word_drop = nn.Dropout(p=0.5)
        self.device = device
        self.emb_dim = emb_dim
        self.topk = topk_keywords
        if dataset == 'Clotho':
            self.sos = 0
            self.eos = 9
        else:
            self.sos = 0
            self.eos = 9
        if pretrain_emb is not None:
            print('Init pretrained word embedding!')
            self.word_emb = EmbeddingC(self.nb_classes, self.emb_dim, pretrain_emb)
        else:
            print('Random word embedding!')
            self.word_emb = nn.Embedding(self.nb_classes, emb_dim, pretrain_emb)
        print("word lens: {}".format(self.nb_classes))
        print('dataset {}'.format(dataset))        
    def init_hidden_state(self, encoder_out, encoder_out1=None, encoder_out2=None):
        if self.multiScale:
            mean_encoder_out = encoder_out.mean(dim=1)
            mean_encoder_out1 = encoder_out1.mean(dim=1)
            mean_encoder_out2 = encoder_out2.mean(dim=1)
            mean_encoder_out = torch.cat((mean_encoder_out, mean_encoder_out1, mean_encoder_out2),-1)
        else:
            mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def GetTag(self, inputs, topk):
        
        batch, device = inputs.shape[0], inputs.device
        TaggingWords = torch.zeros(batch, topk, self.emb_dim).to(device)
        TaggingOneBatch = torch.zeros(batch).long().to(device)
        _, indexes = torch.topk(inputs, topk)
        for ind in range(indexes.shape[1]):
            for i in range(batch):
                TaggingOneBatch[i] = self.tagging_to_embs[indexes[i][ind]]
            tmp = self.word_emb(TaggingOneBatch).detach()
            TaggingWords[:, ind, :] = tmp
        # output = TaggingWords.mean(dim=1)
        output = TaggingWords # bs, topk, dim
        return output


    def trainer(self, x, x1, x2, y, epoch, max_epoch, mixup_lambda=None, classify=None):
        batch_size = x.size(0)
        x = x.transpose(1,2)
        x = self.linear_feature(x) # (B,T,2048 -> 512)
        device = x.device
        if self.multiScale:
            x1 = x1.transpose(1, 2)
            x1 = self.linear_feature1(x1)  # (B,T,128 -> 512)
            x2 = x2.transpose(1, 2)
            x2 = self.linear_feature2(x2)  # (B,T,256 -> 512)
        h, c = self.init_hidden_state(x,x1,x2)
        # x = self.aoa(x)
        # x, _ = self.gru(x,h.unsqueeze(0))
        if self.tag_emb:
            tag_words = self.GetTag(inputs=classify, topk=self.topk)
        predictions = torch.zeros(batch_size, y.shape[1], self.nb_classes).to(device)
        if self.preword_emb:
            previous_wordemb = torch.zeros(batch_size, 1, self.emb_dim,requires_grad=False).to(device) # (B,T,outputdim=512)
        word = self.word_emb(torch.zeros(batch_size).long().to(device))
        word = self.word_drop(word)
        aggregate_att = []
        for t in range(y.shape[1]):
            teacher_focing_ratio = linear_decay(epoch, max_epoch, rate=.7)
            use_teacher_focing = random.random() < teacher_focing_ratio
            attention_weighted_encoding, alpha = self.attention(x, h)
            aggregate_att.append(alpha)
            if self.multiScale:
                attention_weighted_encoding1, alpha1 = self.attention1(x1, h)
                attention_weighted_encoding2, alpha2 = self.attention2(x2, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            if self.multiScale:
                attention_weighted_encoding1 = gate * attention_weighted_encoding1
                attention_weighted_encoding2 = gate * attention_weighted_encoding2
                attention_weighted_encoding = (attention_weighted_encoding + attention_weighted_encoding1 + attention_weighted_encoding2) / 3.
            if self.preword_emb:
                try:
                    prevword_weighted_encoding, beta = self.preword_attention(previous_wordemb,h) # previous time step
                    gate_preword = self.sigmoid(self.f_beta_preword(h))
                    prevword_weighted_encoding = gate_preword * prevword_weighted_encoding
                except:
                    import pdb
                    pdb.set_trace()
            if self.tag_emb:
                taggingword_weighted_encoding, gamma = self.preword_attention(tag_words, h)  # previous time step
                gate_taggingword = self.sigmoid(self.f_beta_taggingword(h))
                taggingword_weighted_encoding = gate_taggingword * taggingword_weighted_encoding

            if self.preword_emb and self.tag_emb:
                h, c = self.decode_step(prevword_weighted_encoding+word+attention_weighted_encoding+taggingword_weighted_encoding,(h, c))
            elif self.preword_emb and not self.tag_emb:
                h, c = self.decode_step(prevword_weighted_encoding + word + attention_weighted_encoding, (h, c))
            elif not self.preword_emb and not self.tag_emb:
                h, c = self.decode_step(word + attention_weighted_encoding, (h, c))

            preds = self.classifier(self.dropout(h))
            predictions[:, t, :] = preds

            if use_teacher_focing and t < y.shape[1]-1:
                word = self.word_emb(y[:,t+1])

            else:
                word = self.word_emb(preds.max(1)[1])

            word = self.word_drop(word)
            p  = word.unsqueeze(1)
            if self.preword_emb:
                previous_wordemb = torch.cat((previous_wordemb,p),1)

        return predictions, aggregate_att
    

    def forward(self,
                x: Tensor, x1: Tensor, x2: Tensor, y: Tensor, epoch :Tensor, max_epoch, mixup_lambda=None, classify=None) \
            -> Tensor:
        """Forward pass of the decoder.
        :param x: Input tensor. Encoder output : (Batch_size, feature_maps, time_steps)
        :param y: Input tensor. Ground Truth : (Batch_size, length)
        :type x: torch.Tensor.
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        
        predictions, aggregate_att = self.trainer(x,x1,x2,y,epoch,max_epoch,mixup_lambda,classify)
                
        return predictions, aggregate_att

    def sample(self, x, x1,x2, y,classify,sample_method):
        device = x.device
        BOS = self.sos 
        EOS = self.eos 
        batch_size = x.size(0)
        # prepare feats, h and c
        x = x.transpose(1, 2)
        x = self.linear_feature(x)  # (B,T,2048 -> 512)
        # x = self.aoa(x)
        if self.multiScale:
            x1 = x1.transpose(1, 2)
            x1 = self.linear_feature1(x1)  # (B,T,128 -> 512)
            x2 = x2.transpose(1, 2)
            x2 = self.linear_feature2(x2)  # (B,T,256 -> 512)
        
        h, c = self.init_hidden_state(x,x1,x2)
        if self.tag_emb:
            tag_words = self.GetTag(inputs=classify, topk=5)
        else:
            tag_words = None


        if self.preword_emb:
            previous_wordemb = torch.zeros(batch_size, 1, self.emb_dim,requires_grad=False).to(self.device) # (B,T,outputdim=512)
        else:
            previous_wordemb = torch.zeros(batch_size, 1, self.emb_dim,requires_grad=False).to(self.device) # (B,T,outputdim=512)
            # None
        it = x.new(batch_size).fill_(BOS).long().to(device)
        seq = x.new_zeros((batch_size, y.shape[1]), dtype=torch.long)
        seqLogprobs = x.new_zeros(batch_size, y.shape[1], self.nb_classes)
        # to collect greedy results
        preds = []
        for t in range(y.shape[1]):
            h, c, it, logprobs = self.step(x, x1, x2, h, c, it, previous_wordemb, tag_words,sample_method)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            word = self.word_emb(it)
            previous_wordemb = torch.cat((previous_wordemb,word.unsqueeze(1)),1)
        return seq, seqLogprobs

    def step(self, x, x1, x2, h, c, it, previous_wordemb=None, tag_word=None, sample_method="normal"):
        attention_weighted_encoding, alpha = self.attention(x, h)
        if self.multiScale:
            attention_weighted_encoding1, alpha1 = self.attention1(x1, h)
            attention_weighted_encoding2, alpha2 = self.attention2(x2, h)
        gate = self.sigmoid(self.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        if self.multiScale:
            attention_weighted_encoding1 = gate * attention_weighted_encoding1
            attention_weighted_encoding2 = gate * attention_weighted_encoding2
            attention_weighted_encoding = (attention_weighted_encoding + attention_weighted_encoding1 + attention_weighted_encoding2) / 3.

        word = self.word_emb(it)
        if self.preword_emb:
            prevword_weighted_encoding, beta = self.preword_attention(previous_wordemb, h)  # previous time step
            gate_preword = self.sigmoid(self.f_beta_preword(h))
            prevword_weighted_encoding = gate_preword * prevword_weighted_encoding
        if self.tag_emb:
            taggingword_weighted_encoding, gamma = self.preword_attention(tag_word, h)  # previous time step
            gate_taggingword = self.sigmoid(self.f_beta_taggingword(h))
            taggingword_weighted_encoding = gate_taggingword * taggingword_weighted_encoding

        if self.preword_emb and self.tag_emb:
            h, c = self.decode_step(
                prevword_weighted_encoding + word + attention_weighted_encoding + taggingword_weighted_encoding, (h, c))
        elif self.preword_emb and not self.tag_emb:
            h, c = self.decode_step(prevword_weighted_encoding + word + attention_weighted_encoding, (h, c))
        elif not self.preword_emb and not self.tag_emb:
            h, c = self.decode_step(word + attention_weighted_encoding, (h, c))

        if sample_method == "normal":
            word_prob = self.classifier(self.dropout(h))
            return h, c, word_prob
        elif sample_method == "greedy":
            word_prob = F.log_softmax(self.classifier(h), dim=-1)
            sampleLogprobs, it = torch.max(word_prob.data, 1)
            it = it.view(-1).long()
            return h,c, it, word_prob
            # return h,c, it, word_prob
        else :
            word_prob = F.log_softmax(self.classifier(h), dim=-1)
            it = torch.distributions.Categorical(logits=word_prob.detach()).sample()
            sampleLogprobs = word_prob.gather(1, it.unsqueeze(1)) # gather the logprobs at sampled positions
            return h,c, it, word_prob
            # return h,c, it, word_prob

    def translate_greedy(self, x, x1, x2, classify=None):
        # BOS = 0
        # EOS = 9
        BOS = self.sos
        EOS = self.eos
        max_len = self.maxlength

        batch_size = x.size(0)
        # prepare feats, h and c
        x = x.transpose(1, 2)
        x = self.linear_feature(x)  # (B,T,2048 -> 512)

        
        # x1, x2 = x, x
        if self.multiScale:
            x1 = x1.transpose(1, 2)
            x1 = self.linear_feature1(x1)  # (B,T,128 -> 512)
            x2 = x2.transpose(1, 2)
            x2 = self.linear_feature2(x2)  # (B,T,256 -> 512)
        
        h, c = self.init_hidden_state(x,x1,x2)
        if self.tag_emb:
            tag_words = self.GetTag(inputs=classify, topk=5)
        else:
            tag_words = None

        it = x.new(batch_size).fill_(BOS).long().to(self.device)
        if self.preword_emb:
            previous_wordemb = torch.zeros(batch_size, 1, self.emb_dim,requires_grad=False).to(self.device) # (B,T,outputdim=512)
        else:
            previous_wordemb = None
        # to collect greedy results
        preds = []

        for t in range(max_len):
            
            h, c, word_prob = self.step(x, x1, x2, h, c, it, previous_wordemb, tag_words)
            # word_prob: [batch_size, vocab_size]

            preds.append(word_prob)
            it = word_prob.max(1)[1]
            word = self.word_emb(it)
            if self.preword_emb:
                previous_wordemb = torch.cat((previous_wordemb,word.unsqueeze(1)),1)
        return torch.stack(preds, dim=1)


    def translate_beam_search(self, x, x1, x2, beam_width=2, alpha=1.15, topk=1, number_required=None, classify=None, usingLM=False):
        if number_required is None:
            number_required = beam_width
        BOS = self.sos
        EOS = self.eos
        max_len = self.maxlength
        if usingLM:
            LMmodel = myGPT()
            LMmodel.eval()

        batch_size, device = x.size(0), x.device
        # prepare feats, h and c
        x = x.transpose(1, 2)
        x = self.linear_feature(x)  # (B,T,2048 -> 512)
        # x1, x2 = x, x 
        if self.multiScale:
            x1 = x1.transpose(1, 2)
            x1 = self.linear_feature1(x1)  # (B,T,128 -> 512)
            x2 = x2.transpose(1, 2)
            x2 = self.linear_feature2(x2)  # (B,T,256 -> 512)
        h, c = self.init_hidden_state(x,x1,x2)
        #refine encoder-out
        # x = self.aoa(x)
        # x, _ = self.gru(x,h.unsqueeze(0))
        if self.tag_emb:
            tag_words = self.GetTag(inputs=classify, topk=5)
        else:
            tag_words = None

        seq_preds = []

        output_dim = h.shape[1]
        # decoding goes sample by sample
        for idx in range(batch_size):
            encoder_output = x[idx, :].unsqueeze(0)
            encoder_output1 = x1[idx, :].unsqueeze(0)
            encoder_output2 = x2[idx, :].unsqueeze(0)
            if self.tag_emb:
                tag_word = tag_words[idx, :].unsqueeze(0)
            else:
                tag_word = None

            endnodes = []
            ## previous word emb
            if self.preword_emb:
                previous_wordemb = torch.zeros(1, 1, self.emb_dim, requires_grad=False).to(self.device) # (B,T,outputdim=512)
            else:
                previous_wordemb = None
            bos = torch.LongTensor([BOS]).to(device)
            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(
                hiddenstate=(h[idx, :].unsqueeze(0), c[idx, :].unsqueeze(0)), 
                previousNode=None, 
                wordId=bos,
                logProb=0, 
                selflp=0, 
                length=0, 
                alpha=alpha,
                previous_wordemb=previous_wordemb,
                all_words=bos
                )
            nodes = PriorityQueue()
            tmp_nodes = PriorityQueue()
            # start the queue
            nodes.put((-node.eval(), node))

            # start beam search
            round = 0

            while True:
                while True:
                    # fetch the best node
                    if nodes.empty(): 
                        break
                    score, n = nodes.get()
                    if (n.wordid[0].cpu().item() == EOS and n.prevNode != None) or (n.leng >= max_len-1):
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required: 
                            break
                        else: 
                            continue

                    # decode for one step using decoder
                    now_h, now_c = n.h
                    it = n.wordid
                    word = self.word_emb(it)

                    new_h, new_c, word_prob = self.step(encoder_output, encoder_output1, encoder_output2, now_h, now_c, it, previous_wordemb, tag_word)
                    word_prob = F.log_softmax(word_prob, dim=-1)

                    if usingLM:
                        output = LMmodel(n.all_words)
                        word_drop_gpt = F.log_softmax(output.logits[-1].unsqueeze(0))
                        word_prob = 0.8 * word_prob + 0.2 * word_drop_gpt

                    # get beam_width candidates
                    log_prob, indexes = torch.topk(word_prob, beam_width)

                    for new_k in range(beam_width):
                        # p_emb = torch.cat((n.previous_wordemb, word.unsqueeze(1)), 1)
                        if self.preword_emb:
                            p_emb = torch.cat((n.previous_wordemb, word.unsqueeze(1)), 1)
                        else:
                            p_emb = None
                        decoded_t = torch.LongTensor([indexes[0][new_k]]).to(device)
                        all_words = torch.cat((n.all_words, decoded_t))
                        log_p = log_prob[0][new_k].item()
                        node = BeamSearchNode((new_h, new_c), n, decoded_t, n.logp, log_p, n.leng + 1, alpha, p_emb,all_words)
                        tmp_nodes.put((-node.eval(), node))

                if len(endnodes) >= number_required or tmp_nodes.empty(): 
                    break
                
                round += 1
                assert nodes.empty()

                # normally, tmp_nodes will have beam_width * beam_width candidates
                # we only keep the most possible beam_width candidates
                for i in range(beam_width):
                    nodes.put(tmp_nodes.get())
                tmp_nodes = PriorityQueue()
                assert tmp_nodes.empty()

            # choose nbest paths, back trace them
            if len(endnodes) < topk:
                for _ in range(topk - len(endnodes)):
                    endnodes.append(nodes.get())

            utterances = []
            count = 1
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                if count > topk: break
                count += 1
                utterance = []

                utterance.append(n.wordid[0].cpu().item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    if n.wordid[0].cpu().item() == BOS: break
                    utterance.append(n.wordid[0].cpu().item())

                # reverse
                utterance = utterance[::-1]
                for i in range(self.maxlength-len(utterance)):
                    utterance.append(EOS)
                utterances.append(utterance)
                
            seq_preds.append(utterances)
            
        seq_preds = np.array(seq_preds)
        seq_preds = torch.from_numpy(seq_preds[:,0,:][:,:,None]).to(device)
        seq_preds = torch.zeros(batch_size, self.maxlength, self.nb_classes).to(device).scatter_(2,seq_preds,1)

        return seq_preds

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, selflp, length, alpha, previous_wordemb, all_words):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        # lstm
        assert isinstance(hiddenstate, tuple)
        self.h = (hiddenstate[0].clone(), hiddenstate[1].clone())

        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb + selflp
        self.selflp = selflp
        self.leng = length
        self.alpha = alpha
        self.previous_wordemb = previous_wordemb
        self.all_words = all_words

    def __lt__(self, other):
        return (-self.eval()) < (-other.eval())

    def eval(self):
        # reward = 0
        # Add here a function for shaping a reward
        # + alpha * reward
        return (self.logp / float(self.leng) ** self.alpha) if self.leng else -1e6


class AttModel(Module):
    def __init__(self,
                 input_dim_encoder: int,
                 hidden_dim_encoder: int,
                 output_dim_encoder: int,
                 embedding_size :int,
                 dropout_p_encoder: float,
                 output_dim_h_decoder: int,
                 nb_classes: int,
                 dropout_p_decoder: float,
                 max_out_t_steps: int,
                 device,
                 encoder_name="tag",
                 pretrain_emb=None,
                 tag_emb=True,
                 multiScale=False,
                 preword_emb=True,
                 two_stage_cnn=True,
                 usingLM=False,
                 topk_keywords=5,
                 dataset='Clotho') \
            -> None:

        super().__init__()
        if encoder_name =="cnn10":
            self.encoder = Cnn10()
        elif encoder_name =="resnet38":
            self.encoder = ResNet38()
        else :
            self.encoder = Tag(500,model_type='resnet38',pretrain_model_path="./models/ResNet38_mAP=0.434.pth",GMAP=False).to(device)

        if two_stage_cnn:
            self.encoder_fixed = Tag(300,model_type='resnet38',pretrain_model_path="./models/ResNet38_mAP=0.434.pth",GMAP=False).to(device)

        self.two_stage_cnn = two_stage_cnn
        self.decoder = AttDecoder(
            input_dim=output_dim_encoder,
            output_dim=output_dim_h_decoder,
            att_dim=output_dim_h_decoder,
            emb_dim = embedding_size,
            maxlength=max_out_t_steps,
            nb_classes=nb_classes,
            dropout_p=dropout_p_decoder,
            device=device,
            pretrain_emb=pretrain_emb,
            tag_emb=tag_emb,
            multiScale = multiScale,
            preword_emb = preword_emb,
            topk_keywords=topk_keywords,
            dataset=dataset
        )
        self.usingLM = usingLM
    
    def forward(self, x, y, epoch, max_epoch,mixup_lambda=None,target_padding_mask=None):
        if self.two_stage_cnn:
            classify, _, _, _ = self.encoder_fixed(x)
            _, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        else:
            classify, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        output = self.decoder(h_encoder, h_encoder1, h_encoder2, y, epoch, max_epoch, mixup_lambda, classify)

        return output, classify
    def encode(self,x):
        classifies, features, _, _  = self.encoder(x)
        return features, classifies

    def decode(self, x, y, epoch, max_epoch,mixup_lambda=None,target_padding_mask=None, classify=None):

        output, aggregate_att = self.decoder(x, None, None, y, epoch, max_epoch, mixup_lambda, classify=classify)
        return output, aggregate_att 
    
    def _sample(self, x, y, sample_method):
        if self.two_stage_cnn:
            classify, _, _, _ = self.encoder_fixed(x)
            _, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        else:
            classify, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        if sample_method == "beamsearch":
            # output = self.decoder.sample(h_encoder, h_encoder1, h_encoder2, y,classify,sample_method)
            output = self.decoder.translate_beam_search(h_encoder, h_encoder1, h_encoder2, beam_width=4, classify=classify, usingLM=self.usingLM)
            output = output.max(2)[1]
        else:
            output = self.decoder.sample(h_encoder, h_encoder1, h_encoder2, y,classify,sample_method)
        return output, classify
    
    def greedy_decode(self,x):
        if self.two_stage_cnn:
            classify, _, _, _ = self.encoder_fixed(x)
            _, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        else:
            classify, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        output = self.decoder.translate_greedy(h_encoder, h_encoder1, h_encoder2, classify)
        output = output.max(2)[1]

        return output, classify

    def beam_search(self,x,beam_width):
        if self.two_stage_cnn:
            classify, _, _, _ = self.encoder_fixed(x)
            _, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        else:
            classify, h_encoder, h_encoder1, h_encoder2 = self.encoder(x)
        output = self.decoder.translate_beam_search(h_encoder, h_encoder1, h_encoder2, beam_width=beam_width, classify=classify, usingLM=self.usingLM)
        output = output.max(2)[1].cpu().numpy().tolist()

        return output

    def freeze_cnn(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        # for p in self.encoder.feature.parameters():
        #     p.requires_grad = False    
    def freeze_classifer(self):
        for p in self.encoder_fixed.parameters():
            p.requires_grad = False
    
    # def preprocess_features_FAT(self, features,FAT,filenames, ratio=0.9): # original
    #     '''
    #     features: (B,C,T)
    #     '''
    #     batch_size = features.shape[0]
    #     num = features.shape[-1]
    #     channel = features.shape[1]
    #     device = features.device
    #     tosample = int(num * ratio)
    #     tomask = num - tosample

    #     # sample_features = torch.zeros(batch_size, channel, tosample).to(device)

    #     sample_features = torch.zeros(batch_size, channel, tosample).to(device)
    #     mask = torch.ones(batch_size,num).to(device)
    #     events = list(FAT.keys())
    #     for i in range(batch_size):
    #         filename = filenames[i]
    #         if filename not in events:
    #             indices = np.random.permutation(num)[:tosample]
    #             indices = sorted(indices)
    #             sample_features[i, :, :] = features[i, :, indices] 
    #         else:
    #             attention_weights = FAT[filename]
    #             # print(attention_weights,attention_weights.shape,tosample)
    #             # print("----------")
    #             # attention_weights = F.softmax(attention_weights,dim=0)
    #             # print(attention_weights)
    #             # print("----------")
    #             # e_num = attention_weights.shape[0]
    #             # e_num = int(e_num*0.1)
    #             idxs = np.arange(num)
    #             # print(num,tosample)
    #             # print(e_num)
    #             indices = torch.multinomial(attention_weights,tomask).cpu().numpy()
    #             # print(indices)
    #             indices_ = list(set(idxs) - set(indices))
    #             # print(indices_,len(indices_))
    #             # print("----------")
    #             indices = sorted(torch.as_tensor(indices_))
    #             # print(indices)
    #             # print("----------")
    #             # print('----end-----')
    #             # print(indices)
    #             sample_features[i, :, :] = features[i, :, indices] 
    #     # print("_________________")
    #     return sample_features
    
    def preprocess_features_FAT(self, features,FAT,filenames, ratio=0.9):
        '''
        features: (B,C,T)
        '''
        batch_size = features.shape[0]
        num = features.shape[-1]
        channel = features.shape[1]
        device = features.device
        tosample = int(num * ratio)
        tomask = num - tosample

        # sample_features = torch.zeros(batch_size, channel, tosample).to(device)

        sample_features = torch.zeros(batch_size, channel, tosample).to(device)
        mask = torch.ones(batch_size,num).to(device)
        # pdb.set_trace()
        if FAT is not None:
            events = list(FAT.keys())
        else:
            events = []
        # fat_batch = [FAT[f] for f in filenames]
        # fat_lens = [len(f) for f in fat_batch]

        all_lens = []
        max_lens = features.shape[-1]
        final_attention_weight = []
        exist_fat = False
        # pdb.set_trace()
        
        for i in range(batch_size):
            filename = filenames[i]
            if filename not in events:
                padding = torch.zeros(max_lens).float().to(device)
                padding += 1
                final_attention_weight.append(padding.unsqueeze(0))
            else:
                attention_weights = FAT[filename].to(device)
                padding = torch.zeros(max_lens - attention_weights.shape[0]).float().to(device)
                
                final_attention_weight.append(torch.cat((attention_weights,padding)).unsqueeze(0))
                exist_fat = True 


        final_attention_weight = torch.cat(final_attention_weight)
        # if not exist_fat:
        #     final_attention_weight += 1.0
        indices = torch.multinomial(final_attention_weight,tosample)
        indices = torch.sort(indices,-1)[0]
        x_ind = torch.arange(0,32).to(device).unsqueeze(1)
        x_ind = x_ind.repeat(1,indices.shape[-1])
        sample_features = features[x_ind,:,indices].transpose(2,1)
        return sample_features
        # for i in range(batch_size):
        #     filename = filenames[i]
        #     if filename not in events:
        #         indices = np.random.permutation(num)[:tosample]
        #         indices = sorted(indices)
        #         sample_features[i, :, :] = features[i, :, indices] 
        #     else:
        #         attention_weights = FAT[filename]
        #         # print(attention_weights,attention_weights.shape,tosample)
        #         # print("----------")
        #         # attention_weights = F.softmax(attention_weights,dim=0)
        #         # print(attention_weights)
        #         # print("----------")
        #         # e_num = attention_weights.shape[0]
        #         # e_num = int(e_num*0.1)
        #         idxs = np.arange(num)
        #         # print(num,tosample)
        #         # print(e_num)
        #         indices = torch.multinomial(attention_weights,tomask).cpu().numpy()
        #         # print(indices)
        #         indices_ = list(set(idxs) - set(indices))
        #         # print(indices_,len(indices_))
        #         # print("----------")
        #         indices = sorted(torch.as_tensor(indices_))
        #         # print(indices)
        #         # print("----------")
        #         # print('----end-----')
        #         # print(indices)
        #         sample_features[i, :, :] = features[i, :, indices] 
        # # print("_________________")
        # return sample_features
    
    def preprocess_feature_random(self, features, ratio):
        batch_size = features.shape[0]
        num = features.shape[-1]
        channel = features.shape[1]
        device = features.device
        tosample = int(num*ratio)
        # tomask = num - tosample

        sample_features = torch.zeros(batch_size, channel, tosample).to(device)
        
        for i in range(batch_size):
            indices = np.random.permutation(num)[:tosample]
            indices = sorted(indices)
            sample_features[i, :, :] = features[i, :, indices] 
        return sample_features
    
## Transformer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000): # max_len=100 2000
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            x = x + self.pe[:x.size(0), :]
        except:
            pdb.set_trace()
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, batch_size, dropout=0.5,pretrain_cnn=None,
                 pretrain_emb=None,freeze_cnn=True, dim_feedforward=2048, use_tags=False, use_threshold=False, threshold=0.4, use_newtrans=False,topk_keywords=5, dataset='Clotho', use_mei=False):
        super(TransformerModel, self).__init__()

        self.model_type = 'cnn+transformer'
        if not use_newtrans:
            from transformer import TransformerDecoder,TransformerDecoderLayer
        else:
            from transformer_idea import TransformerDecoder,TransformerDecoderLayer

        decoder_layers = TransformerDecoderLayer(d_model=nhid, nhead=nhead, dropout=dropout, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.word_emb = nn.Embedding(ntoken, nhid)
        self.ninp = ninp
        self.nhid = nhid
        self.use_tags = use_tags
        self.use_threshold = use_threshold
        self.threshold  = threshold
        self.use_newtrans = use_newtrans
        self.sos = 0
        self.eos = 9
        self.warm_up = True
        self.topk_keywords = topk_keywords
        # self.fc = nn.Linear(512, 512, bias=True)
        # self.fc1 = nn.Linear(512, nhid, bias=True)
        
        if dataset == 'Clotho':
            self.tagging_to_embs = torch.tensor(pickle.load(open(hp.tagging_to_embs, 'rb')))
        else:
            self.tagging_to_embs = torch.tensor(pickle.load(open(hp.tagging_to_embs_audiocaps, 'rb')))
        self.dec_fc = nn.Linear(nhid, ntoken)
        self.batch_size = batch_size
        self.ntoken = ntoken
        # self.encoder = Cnn10()
        if use_mei:
            self.encoder = Tag(500,model_type='Cnn10',pretrain_model_path="./models/Cnn10.pth",GMAP=False)
            self.linear_feature = nn.Linear(512, nhid)
        else:
            self.encoder = Tag(500,model_type='resnet38',pretrain_model_path="./models/ResNet38_mAP=0.434.pth",GMAP=False)
            self.linear_feature = nn.Linear(2048, nhid)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.generator = nn.Softmax(dim=-1)
        
        # self.proj_tags = nn.Linear(nhid, nhid)
        self.init_weights()

        # if pretrain_cnn is not None:
        #     dict_trained = pretrain_cnn
        #     dict_new = self.encoder.state_dict().copy()
        #     new_list = list(self.encoder.state_dict().keys())
        #     trained_list = list(dict_trained.keys())
        #     for i in range(len(new_list)):
        #         dict_new[new_list[i]] = dict_trained[trained_list[i]]
        #     self.encoder.load_state_dict(dict_new)
        if freeze_cnn:
            self.freeze_cnn()

        if pretrain_emb is not None:
            # pdb.set_trace()
            self.word_emb.weight.data[:-1] = pretrain_emb

    def freeze_cnn(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # init_layer(self.fc1)
        # init_layer(self.fc)
        init_layer(self.linear_feature)
        # init_layer(self.proj_tags)
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_fc.bias.data.zero_()
        self.dec_fc.weight.data.uniform_(-initrange, initrange)


    def get_text_emb(self, a_feat, text):
        device = a_feat.device 
        eos_token = self.eos
        # pdb.set_trace()       
        if  self.use_threshold  and not self.warm_up:
            index_x, index_y = torch.where(text>=self.threshold)
            
            index_y = self.tagging_to_embs[index_y].to(device)
            bin_lens = torch.bincount(index_x)
            max_lens = torch.max(bin_lens)
            indexes = torch.ones((a_feat.shape[1],max_lens)).mul(eos_token).long().to(device)
            
            masks = torch.ones((a_feat.shape[1],max_lens)).long().to(device)

            start = 0
            end = 0
            for i, b_lens in enumerate(bin_lens):
                
                if b_lens == 0:
                    # pdb.set_trace()
                    pred_ind = text[i].topk(1)[1]
                    tag_ind = self.tagging_to_embs[pred_ind]
                    indexes[i,0] = tag_ind
                    masks[i][0] = 0
                    continue
                end += b_lens.item()
                y = torch.arange(b_lens).to(device)
                x = torch.ones(b_lens).mul(i).long().to(device)
                
                indexes[x,y] = index_y[start:end]
                masks[x,y] = 0
                
                start = end
                # indexes
            masks = masks.to(bool)
        
        else:
            indexes = text.topk(self.topk_keywords)[1]
            indexes = self.tagging_to_embs[indexes].to(device)
            masks = None
        t_embs = self.word_emb(indexes).transpose(0, 1)
        # projection 
        # t_embs = self.proj_tags(t_embs)
        # pdb.set_trace()
        # t_a_embs = torch.cat((t_embs, a_feat),dim=0)

        # return t_a_embs
        return t_embs, masks

    def encode(self, src, input_mask=None):
        classifies, x, _, _ = self.encoder(src)  # (batch_size, 512, T/16, mel_bins/16)
        # x = torch.mean(x, dim=3)  # (batch_size, 512, T/16)

        x = x.permute(2, 0, 1)  # (T/16,batch_size,512)
        x = self.linear_feature(x)
        if self.use_tags:
            # pdb.set_trace()
            text_embs, text_masks = self.get_text_emb(x, classifies)
            # x = torch.cat((text_embs, x),dim=0)
            # pdb.set_trace()
            return x, text_embs, classifies, text_masks
        else:
            text_embs = None
            text_masks = None 
            return x, text_embs, classifies, text_masks
        # x = F.relu_(self.fc(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = torch.relu(self.fc1(x))
        

    def decode(self, mem_a, mem_t, tgt, input_mask=None, target_mask=None, target_padding_mask=None, epoch=None, text_masks=None, return_weights=False):
        # tgt:(batch_size,T_out)
        # mem:(T_mem,batch_size,nhid)
        
        tgt = tgt.transpose(0, 1)  # (T_out,batch_size)
        # pdb.set_trace()
        if target_mask is None or target_mask.size(0) != len(tgt):
            # pdb.set_trace()
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.dropout(self.word_emb(tgt)) * math.sqrt(self.nhid)
        tgt = self.pos_encoder(tgt)
        # mem = self.pos_encoder(mem)
        
        if self.use_tags:
            # mem_a = torch.cat((mem_t, mem_a),dim=0)
            if not self.use_newtrans:
                output, attn_weights = self.transformer_decoder(tgt, mem_a, memory_mask=input_mask, tgt_mask=target_mask,
                                                tgt_key_padding_mask=target_padding_mask,return_weights=True)
            else:
                output, attn_weights = self.transformer_decoder(tgt, mem_a, mem_t, memory_mask=input_mask, tgt_mask=target_mask,
                                              tgt_key_padding_mask=target_padding_mask, text_key_padding_mask=text_masks,return_weights=True)
        else:
            
            output, attn_weights = self.transformer_decoder(tgt, mem_a, memory_mask=input_mask, tgt_mask=target_mask,
                                            tgt_key_padding_mask=target_padding_mask,return_weights=True)
        
        output = self.dec_fc(output)
        if return_weights:
            return output, attn_weights
        else:
            return output

    def preprocess_features_FAT(self, features,FAT,filenames, ratio=0.9):
        '''
        features: (T,B,C) -> (B,C,T)
        '''
        # pdb.set_trace()
        features = features.permute(1, 2, 0)
        batch_size = features.shape[0]
        num = features.shape[-1]
        channel = features.shape[1]
        device = features.device
        tosample = int(num * ratio)
        tomask = num - tosample

        # sample_features = torch.zeros(batch_size, channel, tosample).to(device)

        # sample_features = torch.zeros(batch_size, channel, tosample).to(device)
        mask = torch.ones(batch_size,num).to(device)
        # pdb.set_trace()
        if FAT is not None:
            events = list(FAT.keys())
        else:
            events = []
        # fat_batch = [FAT[f] for f in filenames]
        # fat_lens = [len(f) for f in fat_batch]

        all_lens = []
        max_lens = features.shape[-1]
        final_attention_weight = []
        exist_fat = False
        # pdb.set_trace()
        
        for i in range(batch_size):
            filename = filenames[i]
            if filename not in events:
                padding = torch.zeros(max_lens).float().to(device)
                padding += 1
                final_attention_weight.append(padding.unsqueeze(0))
            else:
                attention_weights = FAT[filename].to(device)
                padding = torch.zeros(max_lens - attention_weights.shape[0]).float().to(device)
                
                final_attention_weight.append(torch.cat((attention_weights,padding)).unsqueeze(0))
                exist_fat = True 


        final_attention_weight = torch.cat(final_attention_weight)
        # if not exist_fat:
        #     final_attention_weight += 1.0
        indices = torch.multinomial(final_attention_weight,tosample)
        indices = torch.sort(indices,-1)[0]
        x_ind = torch.arange(0,32).to(device).unsqueeze(1)
        x_ind = x_ind.repeat(1,indices.shape[-1])
        sample_features = features[x_ind,:,indices].transpose(2,1)
        sample_features = sample_features.permute(2, 0, 1)
        return sample_features
    
    def forward(self, src, tgt, epoch=None, max_epoch=None, input_mask=None, target_mask=None, target_padding_mask=None):
        # src:(batch_size,T_in,feature_dim)
        # tgt:(batch_size,T_out)
        mem_a, mem_t, classifies, text_masks = self.encode(src)
        output = self.decode(mem_a, mem_t, tgt, input_mask=input_mask, target_mask=target_mask,
                             target_padding_mask=target_padding_mask, text_masks=text_masks)
        return output, classifies

    
    def beam_search(self, x, beam_width=2, alpha=1.15, topk=1, number_required=None, classify=None):
        if number_required is None:
            number_required = beam_width
        BOS = 0
        EOS = 9
        max_len = 30 
        device = x.device

class EnsembleModel(Module):
    def __init__(self,
                 model_list) \
            -> None:

        super().__init__()
        self.model_num = len(model_list)
        self.model_list = model_list

    def greedy_decode(self,x):
        classifys = []
        h_encoders = []
        h_encoder1s = []
        h_encoder2s = []
        for i in range(self.model_num):
            model = self.model_list[i]
            classify, h_encoder, h_encoder1, h_encoder2 = model.encoder(x)
            classifys.append(classify)
            h_encoders.append(h_encoder)
            h_encoder1s.append(h_encoder1)
            h_encoder2s.append(h_encoder2)
        output = self.translate_greedy(h_encoders, h_encoder1s, h_encoder2s, classifys)
        output = output.max(2)[1]

        return output

    def beam_search(self,x,beam_width):
        classifys = []
        h_encoders = []
        h_encoder1s = []
        h_encoder2s = []
        for i in range(self.model_num):
            model = self.model_list[i]
            classify, h_encoder, h_encoder1, h_encoder2 = model.encoder(x)
            classifys.append(classify)
            h_encoders.append(h_encoder)
            h_encoder1s.append(h_encoder1)
            h_encoder2s.append(h_encoder2)
        output = self.translate_beam_search(h_encoders, h_encoder1s, h_encoder2s, beam_width=beam_width, classify=classifys, usingLM=False)
        output = output.max(2)[1].cpu().numpy().tolist()

        return output

    def translate_greedy(self, x, x1, x2, classify):
        
        BOS = 0
        EOS = 9
        h = []
        c = []
        tag_words = []
        previous_wordemb = []
        for i in range(self.model_num):
            max_len = self.model_list[i].decoder.maxlength
            batch_size = x[i].size(0)
            # prepare feats, h and c
            x[i] = x[i].transpose(1, 2)
            x[i] = self.model_list[i].decoder.linear_feature(x[i])  # (B,T,2048 -> 512)
            if self.model_list[i].decoder.multiScale:
                x1[i] = x1[i].transpose(1, 2)
                x1[i] = self.model_list[i].decoder.linear_feature1(x1[i])  # (B,T,128 -> 512)
                x2[i] = x2[i].transpose(1, 2)
                x2[i] = self.model_list[i].decoder.linear_feature2(x2[i])  # (B,T,256 -> 512)
            h_, c_ = self.model_list[i].decoder.init_hidden_state(x[i],x1[i],x2[i])
            h.append(h_)
            c.append(c_)
            if self.model_list[i].decoder.tag_emb:
                tag_words_ = self.model_list[i].decoder.GetTag(inputs=classify[i], topk=5)
            else:
                tag_words_ = None
            tag_words.append(tag_words_)
            it = x[0].new(batch_size).fill_(BOS).long().to(self.model_list[i].decoder.device)
            if self.model_list[i].decoder.preword_emb:
                previous_wordemb_ = torch.zeros(batch_size, 1, self.model_list[i].decoder.emb_dim,requires_grad=False).to(self.model_list[i].decoder.device) # (B,T,outputdim=512)
            else:
                previous_wordemb_ = None
            previous_wordemb.append(previous_wordemb_)

        # to collect greedy results
        preds = []

        for t in range(max_len):
            for i in range(self.model_num):
                h[i], c[i], word_prob_ = self.model_list[i].decoder.step(x[i], x1[i], x2[i], h[i], c[i], it, previous_wordemb[i], tag_words[i])
                if i == 0:
                    word_prob = word_prob_
                else:
                    word_prob = word_prob + word_prob_
            preds.append(word_prob/self.model_num)
            it = word_prob.max(1)[1]
            previous_wordemb_old = previous_wordemb
            previous_wordemb = []
            for i in range(self.model_num):
                word = self.model_list[i].decoder.word_emb(it)
                if self.model_list[i].decoder.preword_emb:
                    previous_wordemb_ = torch.cat((previous_wordemb_old[i],word.unsqueeze(1)),1)
                previous_wordemb.append(previous_wordemb_)
        return torch.stack(preds, dim=1)

    def translate_beam_search(self, x, x1, x2, beam_width=2, alpha=0.7, topk=1, number_required=None, classify=None, usingLM=False):
        if number_required is None:
            number_required = beam_width
        BOS = 0
        EOS = 9
        h = []
        c = []
        tag_words = []
        previous_wordemb = []
        max_len = self.model_list[0].decoder.maxlength
        if usingLM:
            LMmodel = myGPT()
            LMmodel.eval()

        for i in range(self.model_num):
            batch_size, device = x[i].size(0), x[i].device
            # prepare feats, h and c
            x[i] = x[i].transpose(1, 2)
            x[i] = self.model_list[i].decoder.linear_feature(x[i])  # (B,T,2048 -> 512)
            if self.model_list[i].decoder.multiScale:
                x1[i] = x1[i].transpose(1, 2)
                x1[i] = self.model_list[i].decoder.linear_feature1(x1[i])  # (B,T,128 -> 512)
                x2[i] = x2[i].transpose(1, 2)
                x2[i] = self.model_list[i].decoder.linear_feature2(x2[i])  # (B,T,256 -> 512)
            h_, c_ = self.model_list[i].decoder.init_hidden_state(x[i],x1[i],x2[i])
            h.append(h_[None,:])
            c.append(c_[None,:])
            if self.model_list[i].decoder.tag_emb:
                tag_words_ = self.model_list[i].decoder.GetTag(inputs=classify[i], topk=5)
            else:
                tag_words_ = None
            tag_words.append(tag_words_)
            ## previous word emb
            if self.model_list[i].decoder.preword_emb:
                previous_wordemb_ = torch.zeros(1, 1, self.model_list[i].decoder.emb_dim, requires_grad=False).to(self.model_list[i].decoder.device) # (B,T,outputdim=512)
                
            else:
                previous_wordemb_ = torch.zeros(1, 1, self.model_list[i].decoder.emb_dim, requires_grad=False).to(self.model_list[i].decoder.device)
            previous_wordemb.append(previous_wordemb_[None,:])

        h = torch.cat(h, 0)
        c = torch.cat(c, 0)
       
        previous_wordemb = torch.cat(previous_wordemb,0)

        seq_preds = []

        output_dim = h[0].shape[1]
        # decoding goes sample by sample
        for idx in range(batch_size):
            endnodes = []
            bos = torch.LongTensor([BOS]).to(device)
            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(
                hiddenstate=(h[:, idx].unsqueeze(1), c[:, idx].unsqueeze(1)), 
                previousNode=None, 
                wordId=bos,
                logProb=0, 
                selflp=0, 
                length=0, 
                alpha=alpha,
                previous_wordemb=previous_wordemb,
                all_words=bos
                )
            nodes = PriorityQueue()
            tmp_nodes = PriorityQueue()
            # start the queue
            nodes.put((-node.eval(), node))

            # start beam search
            round = 0

            while True:
                while True:
                    # fetch the best node
                    if nodes.empty(): 
                        break
                    score, n = nodes.get()
                    if (n.wordid[0].cpu().item() == EOS and n.prevNode != None) or (n.leng >= max_len-1):
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required: 
                            break
                        else: 
                            continue

                    # decode for one step using decoder
                    now_h, now_c = n.h
                    it = n.wordid
                    new_h, new_c = [], []
                    for i in range(self.model_num):
                        encoder_output = x[i]
                        encoder_output = encoder_output[idx, :].unsqueeze(0)
                        encoder_output1 = x1[i]
                        encoder_output1 = encoder_output1[idx, :].unsqueeze(0)
                        encoder_output2 = x2[i]
                        encoder_output2 = encoder_output2[idx, :].unsqueeze(0)
                        if self.model_list[i].decoder.tag_emb:
                            tag_word = tag_words[i]
                            tag_word = tag_word[idx, :].unsqueeze(0)
                        else:
                            tag_word = None
                        new_h_, new_c_, word_prob_ = self.model_list[i].decoder.step(encoder_output, encoder_output1, encoder_output2, now_h[i], now_c[i], it, previous_wordemb[i], tag_word)
                        word_prob_ = F.log_softmax(word_prob_, dim=-1)
                        new_h.append(new_h_[None,:])
                        new_c.append(new_c_[None,:])
                        if i == 0:
                            word_prob = word_prob_
                        else:
                            word_prob = word_prob + word_prob_
                    new_h = torch.cat(new_h, 0)
                    new_c = torch.cat(new_c, 0)
                    word_prob = word_prob/self.model_num
                    if usingLM:
                        output = LMmodel(n.all_words)
                        word_drop_gpt = F.log_softmax(output.logits[-1].unsqueeze(0))
                        word_prob = 0.9 * word_prob + 0.1 * word_drop_gpt

                    # get beam_width candidates
                    log_prob, indexes = torch.topk(word_prob, beam_width)

                    for new_k in range(beam_width):
                        p_emb = []
                        for i in range(self.model_num):
                            word = self.model_list[i].decoder.word_emb(it)
                            if self.model_list[i].decoder.preword_emb:
                                p_emb_ = torch.cat((n.previous_wordemb[i],word.unsqueeze(1)),1)
                            else:
                                p_emb_ = None
                            p_emb.append(p_emb_)
                        decoded_t = torch.LongTensor([indexes[0][new_k]]).to(device)
                        all_words = torch.cat((n.all_words, decoded_t))
                        log_p = log_prob[0][new_k].item()
                        node = BeamSearchNode((new_h, new_c), n, decoded_t, n.logp, log_p, n.leng + 1, alpha, p_emb,all_words)
                        tmp_nodes.put((-node.eval(), node))

                if len(endnodes) >= number_required or tmp_nodes.empty(): 
                    break
                
                round += 1
                assert nodes.empty()

                # normally, tmp_nodes will have beam_width * beam_width candidates
                # we only keep the most possible beam_width candidates
                for i in range(beam_width):
                    nodes.put(tmp_nodes.get())
                tmp_nodes = PriorityQueue()
                assert tmp_nodes.empty()

            # choose nbest paths, back trace them
            if len(endnodes) < topk:
                for _ in range(topk - len(endnodes)):
                    endnodes.append(nodes.get())

            utterances = []
            count = 1
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                if count > topk: break
                count += 1
                utterance = []

                utterance.append(n.wordid[0].cpu().item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    if n.wordid[0].cpu().item() == BOS: break
                    utterance.append(n.wordid[0].cpu().item())

                # reverse
                utterance = utterance[::-1]
                for i in range(self.model_list[0].decoder.maxlength-len(utterance)):
                    utterance.append(EOS)
                utterances.append(utterance)
                # print(utterances)
            seq_preds.append(utterances)
            
        seq_preds = np.array(seq_preds)
        seq_preds = torch.from_numpy(seq_preds[:,0,:][:,:,None]).to(device)
        seq_preds = torch.zeros(batch_size, self.model_list[0].decoder.maxlength, self.model_list[0].decoder.nb_classes).to(device).scatter_(2,seq_preds,1)

        return seq_preds

if __name__ == '__main__':
    from hparams import hparams 
    hp = hparams()
    device = torch.device('cuda:0')
    pretrain_cnn = torch.load(hp.pretrain_cnn_path,map_location='cuda:0')
    # model = AttModel(hp.ntoken, hp.ninp, hp.nhead, hp.nhid, hp.nhid, hp.nlayers, hp.batch_size, max_len=30, device=device, model_type='resnet38',dropout=0.2,
    #                          pretrain_cnn=pretrain_cnn, pretrain_emb=None, freeze_cnn=True).to(device)
    model =  AttModel(hp.output_dim_encoder,hp.output_dim_h_decoder,hp.att_size,hp.emb_size,hp.max_out_t_steps,hp.ntoken,hp.dropout_p_decoder).to(device)
    inputs = torch.rand(1, 2000, 64).to(device)
    tag = torch.ones(1, 10).long().to(device)
    y = 1
    y_ = 30
    print(model(inputs, tag, y, y_).shape)
