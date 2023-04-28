import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from data_handling import get_clotho_loader, get_test_data_loader
# from model import AttModel, TransformerModel # , RNNModel, RNNModelSmall
from model_visualization import AttModel, TransformerModel
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
from scripts.scst import scst_loss,RewardCriterion,get_self_critical_reward,get_self_cider
from coco_caption.pycocoevalcap.cider.cider import Cider
import pickle
from warmup_scheduler import GradualWarmupScheduler
scorer = Cider()

hp = hparams()
parser = argparse.ArgumentParser(description='hparams for model')


np.random.seed(hp.seed)
torch.manual_seed(hp.seed)
random.seed(hp.seed)

def clip_bce(output_dict, target_dict):
    return F.binary_cross_entropy_with_logits(
        output_dict, target_dict)
from torch.nn.modules.loss import _WeightedLoss
## 继承_WeightedLoss类
class SmoothingBCELoss(_WeightedLoss):
	def __init__(self, weight=None, reduction='mean', smoothing=0.0):
		super(SmoothingBCELoss, self).__init__(weight=weight, reduction=reduction)
		self.smoothing = smoothing
		self.weight  = weight
		self.reduction = reduction
	# @staticmethod
	def _smooth(self,targets, n_labels, smoothing=0.0):
		assert 0 <= smoothing < 1
		with torch.no_grad():
			targets = targets  * (1 - smoothing) + 0.5 * smoothing
		return targets
	def forward(self, inputs, targets):
		# pdb.set_trace()
		targets = self._smooth(targets, inputs.size(-1), self.smoothing)
		loss = F.binary_cross_entropy_with_logits(inputs, targets)
		
		if self.reduction == 'sum':
			loss = loss.item()
		elif self.reduction == 'mean':
			loss = loss.mean()
		return loss
		
clip_bce = SmoothingBCELoss(smoothing=0.01)
def train(epoch, max_epoch, mixup=False, augmentation=None):
    model.train()
    total_loss_text = 0.
    start_time = time.time()
    batch = 0
    return_loss = []
    loss_tags = 0
    # if epoch >= 5:
    #     ratio_t = 0.1
    #     ratio_a = 1.
    # else:
    #     ratio_t = 1
    #     ratio_a = 0.01
    ratio_t = 1
    ratio_a = 0.5
    with torch.autograd.set_detect_anomaly(True):
        for src, tgt, tgt_len, ref, filename, tags, _ in training_data:
            tgt_y = tgt[:, 1:]
            if mixup:
                mixup_lambda = augmentation.get_lambda(src.shape[0])
                tgt_y = tgt2onehot(tgt_y,hp.ntoken)
                src = do_mixup(src,mixup_lambda=mixup_lambda).float()
                tgt_y = do_mixup(tgt_y,mixup_lambda=mixup_lambda).float()
            src = src.to(device)
            tgt = tgt.to(device)
            tags = tags.to(device).float()
            tgt_pad_mask = get_padding(tgt, tgt_len)
            tgt_in = tgt[:, :-1]
            tgt_pad_mask = tgt_pad_mask[:, :-1]
            
            tgt_y = tgt_y.to(device)
            optimizer.zero_grad()
            output,classifies = model(src, tgt_in, epoch, max_epoch, target_padding_mask=tgt_pad_mask)
            
            
            # No padding for loss calculation
            if hp.NopadLoss:
                if hp.decoder == 'Transformer' :
                    output = output.transpose(0, 1)
                tgt_y_ = torch.cat([tgt_y[j][:tgt_len[j]-1] for j in range(tgt_y.shape[0])], 0)
                output_ = torch.cat([output[j][:tgt_len[j]-1] for j in range(tgt_y.shape[0])], 0)
            else:
                tgt_y_ = tgt_y
                output_ = output

            if mixup:
                loss_text = clip_bce(output_, tgt_y_)
            else:
                loss_text = criterion(output_.contiguous().view(-1, hp.ntoken), tgt_y_.contiguous().view(-1))
            if hp.use_tags_loss:
                loss_tag = clip_bce(classifies, tags)
            
                loss = ratio_t * loss_text + ratio_a * loss_tag
                loss_tags += loss_tag.item()
            else:
                loss = loss_text
            # pdb.set_trace()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
            optimizer.step()
            total_loss_text += loss_text.item()
            return_loss.append(loss_text.item())

            
            writer.add_scalar('Loss/train-text', loss_text.item(), (epoch - 1) * len(training_data) + batch)

            batch += 1

            if batch % hp.log_interval == 0 and batch > 0:
                mean_text_loss = total_loss_text / hp.log_interval
                mean_loss_tags = loss_tags / hp.log_interval
                elapsed = time.time() - start_time
                current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                            'loss-text {:5.4f} | loss-tag {:5.4f} '.format(
                    epoch, batch, len(training_data), current_lr,
                    elapsed * 1000 / hp.log_interval, mean_text_loss, mean_loss_tags))
                
                total_loss_text = 0
                loss_tags = 0.
                start_time = time.time()
            # print(batch)
    return np.mean(return_loss)

def eval_all(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None, eval_model='Transformer'):
    model.eval()

    with torch.no_grad():
        output_sentence_all = []
        ref_all = []

        pred_output = []
        all_tag = []
        loss_eval = []
       
        for src, tgt, _, ref, filename,tags, _ in evaluation_data:
            src = src.to(device)
            tags = tags.to(device).float()
            
            if eval_model == 'Transformer':
                output, classifies = greedy_decode(model, src, max_len=max_len)
                
            else:
                output, classifies = model.greedy_decode(src)

            loss_tag = clip_bce(classifies, tags)
            classifies = torch.sigmoid(classifies)
            pred_output.extend(classifies.cpu().numpy())
            all_tag.extend(tags.cpu().numpy())
            loss_eval.append(loss_tag.item())
            
            # pdb.set_trace()
            output_sentence_ind_batch = []
            for i in range(output.size()[0]):
                output_sentence_ind = []
                for j in range(1, output.size(1)):
                    sym = output[i, j]
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)
        # pdb.set_trace()
        # print(len(ref_all))
        pred_output = np.array(pred_output)
        all_tag = np.array(all_tag)
        mean_loss = np.mean(loss_eval)
        average_precision = metrics.average_precision_score(
                all_tag, pred_output, average=None)
        average_precision= average_precision[~np.isnan(average_precision)]
        # pdb.set_trace()
        score, output_str, ref_str, all_score = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)
        
        loss_mean = score
        c_score = all_score['CIDEr']
        writer.add_scalar(f'Loss/eval_greddy', loss_mean, epoch)
        # pdb.set_trace()
        msg = f'eval_greddy SPIDEr: {loss_mean:2.4f} | loss_tag: {mean_loss:2.5f} | mAP: {np.mean(average_precision):2.5f} | CIDEr: {c_score:2.4f}'
        logging.info(msg)
        
def eval_with_beam(evaluation_data, max_len=30, sos_ind=0, eos_ind=9, word_dict_pickle_path=None, beam_size=3, eval_model='Transformer'):
    model.eval()
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    cider_scorer = CiderScorer(df="scripts/output_pkl_inds.p")
    
    # filenames = ['07 ambient bell.wav','080809_05_FontanaKoblerov.wav','20080226.serins.rbd.02.wav','20100422.waterfall.birds.wav']
    with torch.no_grad():
        
        output_sentence_all = []
        ref_all = []
        for i, (src, tgt, _, ref,filename,tags, _) in enumerate(evaluation_data):
            # output_sentence_all = []
            # ref_all = []
            src = src.to(device)

            if eval_model == 'Transformer':
                # output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)
                output = beam_decode(src, model, sos_ind=sos_ind, eos_ind=eos_ind, beam_width=beam_size, kwp=hp.use_tags_t)
            else:
                output = model.beam_search(src, beam_width=beam_size)
            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break    
                    if eval_model == 'Transformer':
                        output_sentence_ind.append(sym.item())
                    else:
                        output_sentence_ind.append(sym)
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)
        
        score, output_str, ref_str, all_score = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        # pdb.set_trace()
        c_score = all_score['CIDEr']
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size}  SPIDEr: {loss_mean:2.4f}  CIDEr: {c_score:2.4f} allscore {all_score} '
        logging.info(msg)
        

def test_with_beam(test_data, max_len=30, eos_ind=9, beam_size=3,eval_model='Transformer',name="seed1111"):
    model.eval()

    with torch.no_grad():
        save_name  = "test_out_" + name + ".csv"
        with open(save_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'caption_predicted'])
            for src, filename in test_data:
                src = src.to(device)
                if eval_model == 'Transformer':
                    output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)
                else:
                    output = model.beam_search(src, beam_width=beam_size)
                output_sentence_ind_batch = []
                for single_sample in output:
                    output_sentence_ind = []
                    for sym in single_sample:
                        if sym == eos_ind: break
                        if eval_model == 'Transformer':
                            output_sentence_ind.append(sym.item())
                        else:
                            output_sentence_ind.append(sym)
                    output_sentence_ind_batch.append(output_sentence_ind)
                out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
                for caption, fn in zip(out_str, filename):
                    writer.writerow(['{}.wav'.format(fn), caption])

def eval_with_beam_csv(evaluation_data, max_len=30, sos_ind=0, eos_ind=9, word_dict_pickle_path=None, beam_size=3, eval_model='Transformer'):
    model.eval()
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    with open('./audio_tag/word_list_500_train.p', 'rb') as f:
        testtag = np.array(pickle.load(f))

    with torch.no_grad():
        with open("visual_output/attmodel_new_out.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['filename','caption_groudtruth','caption_predicted','cider','tags'])
            for src, tgt, _, ref,filename,tags, _ in evaluation_data:
                # print(src.shape)
                src = src.to(device)
                # _, _, classifications, _ = model.encode(src)
                classifications, _, _, _ = model.encoder(src)
                classifications = classifications.topk(5)[1].cpu()
                classifications = testtag[classifications].tolist()
                tgt = tgt.numpy().tolist()
                if eval_model == 'Transformer':
                    output = beam_decode(src, model, sos_ind=sos_ind, eos_ind=eos_ind, beam_width=beam_size, kwp=hp.use_tags_t)
                else:
                    output = model.beam_search(src, beam_width=beam_size)
                output_sentence_ind_batch = []
                for single_sample in output:
                    output_sentence_ind = []
                    for sym in single_sample:
                        if sym == eos_ind: break
                        if eval_model == 'Transformer':
                            output_sentence_ind.append(sym.item())
                        else:
                            output_sentence_ind.append(sym)
                    output_sentence_ind_batch.append(output_sentence_ind)
                # pdb.set_trace()
                cider_score = get_self_cider(output_sentence_ind_batch, ref, word_dict, scorer)
                cider_score = np.round(cider_score,3).tolist()
                out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
                _ , ref_str = get_eval(tgt, ref, hp.word_dict_pickle_path)
                
                # print(ref_str[0])
                for caption, groundtruth,fname, c_score, t_class in zip(out_str, ref_str,filename, cider_score, classifications):
                    writer.writerow([fname,groundtruth,caption,t_class, c_score])

def save_pic(one_dict):
    import librosa 
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    import cv2
    from PIL import Image
    out_str = one_dict['out_str']
    attend_a = one_dict['attend_a']
    attend_t = one_dict['attend_t']
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    path = "/data/yzj_data/workspace/DCASE2021_Task6_PKU/create_dataset/data/clotho_audio_files/evaluation/Flowing traffic in the outer ring of Milan 2.wav"

    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(path, sr=None, mono=False)
    print(fs)
    L = len(data)
    print('Time:', L / fs)
    # 0.025s
    framelength = 1024/fs
    # NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    print("NFFT:", framesize)
    #提取mel特征
    mel_spect = librosa.feature.melspectrogram(data, sr=fs, n_fft=framesize,n_mels=64)
    #转化为log形式
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    #画mel谱图
    librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
    pylab.savefig('t_output/1.png', bbox_inches=None, pad_inches=0)
    pylab.close()
    for i, att in enumerate(attend_a[:-1]):
        att = att.numpy()[:,np.newaxis,:]
        mask = att.repeat(64,axis=1)
        mask = mask.transpose(2,1,0)
        # print(att.shape)
        mask = cv2.resize(mask,(288,432))
        # heatmap = mask/np.max(mask)
        # heatmap = np.uint8(255 * heatmap)
        mask = mask.transpose(1,0)
        img = Image.open("t_output/1.png")
        # plt.imshow(mask, alpha=0.4, cmap='gray')  #alpha设置透明度, cmap可以选择颜色

        # img = img + mask
        pylab.imshow(img)
        pylab.imshow(mask, alpha=0.6,cmap='gray' )
        pylab.savefig('t_output/1_out_{}.png'.format(out_str[i]), bbox_inches=None, pad_inches=0)
        # pylab.show()
        # break 

def plot_visual(attention_weight, x_ind, y_ind, save_dir):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.size'] = 14
    # plt.rcParams['xtick.visible'] = False
    # plt.rcParams['ytick.visible'] = False
    # plt.rcParams['axes.spines.top'] = False
    # plt.rcParams['axes.spines.right'] = False
    # attention_weight = torch.cat(attention_weight).transpose(0,1).numpy()
    sns.set()
    plt.figure(figsize=(10, 5))
    sns.heatmap(attention_weight,  cmap='Blues', annot=False,cbar=False)
    xticklabels =  x_ind
    yticklabels =  y_ind
    plt.xticks(np.arange(len(xticklabels))+0.5, xticklabels, fontsize=12, rotation=90)
    plt.yticks(np.arange(len(yticklabels))+0.5, yticklabels, fontsize=12,rotation=0)
    # plt.colorbar(cax=None)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir,dpi=400)

def eval_with_beam_visualization(evaluation_data, max_len=30, sos_ind=0, eos_ind=9, word_dict_pickle_path=None, beam_size=3, eval_model='Transformer'):
    model.eval()
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'

    tagwords = get_word_dict('audio_tag/word_list_500_train.p')
    
    # filenames = ['07 ambient bell.wav','080809_05_FontanaKoblerov.wav','20080226.serins.rbd.02.wav','20100422.waterfall.birds.wav']
    with torch.no_grad():
        # model.decoder.visualization=True
        output_sentence_all = []
        ref_all = []
        for i, (src, tgt, _, ref,filename,tags, lens) in enumerate(evaluation_data):
            # output_sentence_all = []
            # ref_all = []
            # pdb.set_trace()
            # if filename[0] != 'night ambient crickets bugs white noise occasional cough.wav':
            #     continue
            # f_name = 'door-squeak-rattle.wav'
            # f_name = 'Tenerife_bazaar_2.wav'
            f_name = 'Wind Chimes On Town Square, Germany.wav'
            # f_name = 'Flowing traffic in the outer ring of Milan 2.wav'
            # f_name = 'Brushing_Teeth_Bathroom_Fx.wav'
            # f_name = 'Dogs barking from barn in distance in the morning.wav'
            # f_name = 'Tenerife_bazaar_2.wav'
            if f_name not in filename:
                continue
            # pdb.set_trace()
            src = src.to(device)

            if eval_model == 'Transformer':
                # output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)
                output, classify, attend_a, attend_t = beam_decode(src, model, sos_ind=sos_ind, eos_ind=eos_ind, beam_width=beam_size, kwp=hp.use_tags_t, visualization=True)
            else:
                output, attend_a, attend_t, classify = model.beam_search(src, beam_width=beam_size)
            # pdb.set_trace()
            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break    
                    if eval_model == 'Transformer':
                        output_sentence_ind.append(sym.item())
                    else:
                        output_sentence_ind.append(sym)
                output_sentence_ind_batch.append(output_sentence_ind)
            # output_sentence_all.extend(output_sentence_ind_batch)
            # ref_all.extend(ref)
            
            
            out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
            cider_score = get_self_cider(output_sentence_ind_batch, ref, word_dict, scorer)
            cider_score = np.round(cider_score,3).tolist()
            new_dict = {}
            #
            if  eval_model == 'Transformer':
                # pdb.set_trace()
                index = filename.index(f_name)
                tags_index = classify[index].topk(5)[1].cpu().numpy()
                tags_file = [tagwords[ind] for ind in tags_index]
                cur_attend_a = attend_a[index]
                cur_attend_t = attend_t[index]
                for i in range(len(cur_attend_a)):
                    att_a = cur_attend_a[i][0].transpose(0,1).numpy()
                    att_t = cur_attend_t[i][0].transpose(0,1).numpy()
                    x_ind_str = out_str[index].split()
                    x_ind_str.append('<eos>')
                    # pdb.set_trace()
                    lens_ = lens[index]*512/44100 /10
                    x_index = np.array(' ', dtype='U100').repeat(att_a.shape[0])
                    lens_y = att_a.shape[0]/lens_
                    for k in range(int(np.ceil(lens_))):

                        x_index[int(lens_y * k)]=str(k*10)
                    # pdb.set_trace()
                    plot_visual(att_t, x_ind_str, tags_file, 't_output/{}_tag_trans_decoder_new{}.png'.format(f_name,i))
                    plot_visual(att_a, x_ind_str, x_index, 't_output/{}_audio_trans_decoder_new{}.png'.format(f_name,i))
            else:
                index = filename.index(f_name)
                tags_index = classify[index].topk(5)[1].cpu().numpy()
                tags_file = [tagwords[ind] for ind in tags_index]
                new_dict['fname'] = f_name
                new_dict['out_str'] = out_str[index]
                new_dict['attend_a']=attend_a[index]
                new_dict['attend_t']=attend_t[index]
                new_dict['tags']=tags_file
                
                x_ind_str = out_str[index].split()
                attend_a = torch.cat(attend_a[index]).transpose(0,1).numpy()
                attend_t = torch.cat(attend_t[index]).transpose(0,1).numpy()
                # x_ind_str.insert(0,'<sos>')
                x_ind_str.append('<eos>')
                
                lens = lens[index]*512/44100 /10
                x_index = np.array(' ', dtype='U100').repeat(attend_a.shape[0])
                lens_y = attend_a.shape[0]/lens
                for i in range(int(np.ceil(lens))):

                    x_index[int(lens_y * i)]=str(i*10)
                # pdb.set_trace()
                plot_visual(attend_t, x_ind_str, tags_file, 't_output/{}_tag_trans.png'.format(f_name))
                plot_visual(attend_a, x_ind_str, x_index, 't_output/{}_audio_trans.png'.format(f_name))
            # with open('t_output/one.pickle', 'wb') as f:
            #     pickle.dump(new_dict, f)
            # save_pic(new_dict)
            # out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
            # _ , ref_str = get_eval(tgt, ref, hp.word_dict_pickle_path)
            pdb.set_trace()
        score, output_str, ref_str, all_score = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        # pdb.set_trace()
        c_score = all_score['CIDEr']
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size}  SPIDEr: {loss_mean:2.4f}  CIDEr: {c_score:2.4f} allscore {all_score} '
        logging.info(msg)

if __name__ == '__main__':
    parser.add_argument('--device', type=str, default=hp.device)
    parser.add_argument('--nlayers', type=int, default=hp.nlayers)
    parser.add_argument('--nhead', type=int, default=hp.nhead)
    parser.add_argument('--nhid', type=int, default=hp.nhid)
    parser.add_argument('--batch_size', type=int, default=hp.batch_size)
    parser.add_argument('--training_epochs', type=int, default=hp.training_epochs)
    parser.add_argument('--lr', type=float, default=hp.lr)
    parser.add_argument('--scheduler_decay', type=float, default=hp.scheduler_decay)
    parser.add_argument('--load_pretrain_cnn', action='store_true')
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument('--load_pretrain_emb', action='store_true')
    parser.add_argument('--load_pretrain_model', action='store_true')
    parser.add_argument('--spec_augmentation', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--name', type=str, default=hp.name)
    parser.add_argument('--mode', type=str, default=hp.mode)
    parser.add_argument('--eval_dir', type=str, default=hp.eval_dir)
    parser.add_argument('--pretrain_emb_path', type=str, default=hp.pretrain_emb_path)
    parser.add_argument('--pretrain_cnn_path', type=str, default=hp.pretrain_cnn_path)
    parser.add_argument('--pretrain_model_path', type=str, default=hp.pretrain_model_path)
    parser.add_argument('--decoder', type=str, default=hp.decoder)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--finetune_ce', action='store_true')
    parser.add_argument('--use_tags_loss', action='store_true')

    
    # att model 
    parser.add_argument('--tag_emb', action='store_true')
    parser.add_argument('--preword_emb', action='store_true')
    parser.add_argument('--dataset', type=str, default='Clotho')

    # transformer model
    parser.add_argument('--use_tags_t', action='store_true')
    parser.add_argument('--use_threshold_t', action='store_true')
    parser.add_argument('--use_newtrans', action='store_true')
    parser.add_argument('--threshold_t', type=float, default=hp.threshold_t)
    parser.add_argument('--encoder_lr', type=float, default=hp.encoder_lr)
    parser.add_argument('--topk_keywords', type=int, default=hp.topk_keywords)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(hp, k, v)
    args = parser.parse_args()
    # pdb.set_trace()
    if hp.dataset == 'AudioCaps':
        hp.ntoken = 5046+1
    device = torch.device(hp.device)
    eval_model = hp.decoder

    if hp.load_pretrain_emb:
        pretrain_emb = pickle.load(open(hp.pretrain_emb_path,'rb'))
    else:
        pretrain_emb = None
    pretrain_cnn = torch.load(hp.pretrain_cnn_path, map_location="cpu") if hp.load_pretrain_cnn else None
    
    if hp.decoder == 'AttDecoder':
        if hp.load_pretrain_emb:
            print("load pretrain embedding")
        model = AttModel(hp.ninp,hp.nhid,hp.output_dim_encoder,hp.emb_size,hp.dropout_p_encoder,
        hp.output_dim_h_decoder,hp.ntoken,hp.dropout_p_decoder,hp.max_out_t_steps,device,'tag',pretrain_emb,hp.tag_emb,
        hp.multiScale,hp.preword_emb,hp.two_stage_cnn,hp.usingLM).to(device)
        
    elif hp.decoder == 'Transformer': # no used now
        model = TransformerModel(hp.ntoken, hp.ninp, hp.nhead_t, hp.nhid_t, hp.nlayers_t, hp.batch_size, dropout=0.2,
                             pretrain_cnn=pretrain_cnn, pretrain_emb=pretrain_emb, freeze_cnn=hp.freeze_cnn, dim_feedforward=hp.dim_feedforward, use_tags=hp.use_tags_t, 
                             use_threshold=hp.use_threshold_t, threshold=hp.threshold_t,use_newtrans=hp.use_newtrans).to(device)
    else :
        print('exit!!!')
        sys.exit(0)
    ## load pretrain model 
    # pdb.set_trace()
    print("The model is", hp.decoder)
    if pretrain_cnn is not None:
        dict_trained = pretrain_cnn
        dict_new = model.encoder.state_dict().copy()
        new_list = list(model.encoder.state_dict().keys())
        trained_list = list(dict_trained.keys())
        for i in range(len(new_list)):
            print(new_list[i])
            dict_new[new_list[i]] = dict_trained[trained_list[i]]
        model.encoder.load_state_dict(dict_new)
        if hp.two_stage_cnn:
            model.encoder_fixed.load_state_dict(dict_new)

    if hp.load_pretrain_model:
        model.load_state_dict(torch.load(hp.pretrain_model_path,map_location="cpu"))
    print("freeze_cnn", hp.freeze_cnn)
    if hp.freeze_cnn:
        model.freeze_cnn()
        print("freeze_cnn has finished!")
    if hp.freeze_classifer and hp.two_stage_cnn:
        model.freeze_classifer()
        print("freeze_classifer has finished!")

    param_dicts = [
                {"params": [p for n, p in model.named_parameters() if
                            "encoder" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if
                               "encoder" in n and "encoder.fc" not in n and  p.requires_grad],
                    "lr": hp.encoder_lr,   # 3e-5
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               "encoder.fc" in n and  p.requires_grad],
                    "lr": 1e-3,   # 3e-5
                },

            ]
    

    optimizer = torch.optim.Adam(params=param_dicts, lr=hp.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    if hp.label_smoothing:
        criterion = LabelSmoothingLoss(hp.ntoken, smoothing=0.1, word_freq=hp.word_freq_reciprocal_pickle_path)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=hp.ntoken - 1)
    if hp.multi_gpu:
        device_ids = [4,5]
        # model.to(device)
        model = torch.nn.DataParallel(model,device_ids=device_ids)

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    # log_dir = 'models_trans/{name}'.format(name=hp.name)
    log_dir = hp.name
    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

    if args.resume:
        print("use resume")
        checkpoint = torch.load(args.resume,map_location="cpu")
        # pdb.set_trace()
        model.load_state_dict(checkpoint['model'])
        if args.finetune_ce:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr, weight_decay=1e-6)
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.2)
            epoch = 1
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch = checkpoint['epoch']
            model.warm_up = False
            epoch += 1
    else:
        print("train from scratch")
        epoch = 1

    
    logging.basicConfig(level=logging.DEBUG,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)]
                        )
    if hp.dataset == 'AudioCaps':
        data_dir = hp.data_dir_audiocaps
        word_dict_pickle_path = hp.word_dict_pickle_path_audiocaps
        sos_ind = 5044
        eos_ind = 5045

       
    else:
        data_dir = hp.data_dir
        word_dict_pickle_path = hp.word_dict_pickle_path
        sos_ind = 0
        eos_ind = 9

    word_freq_pickle_path = hp.word_freq_pickle_path

    # eval_data_dir = hp.eval_data_dir
    # train_data_dir = hp.train_data_dir
    

    test_data_dir = hp.test_data_dir
    num_workers = 4
    if hp.train_all:
        training_data = get_clotho_loader(data_dir=data_dir, split='all',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=4, return_reference=True, augment=hp.spec_augmentation)
    else:
        training_data = get_clotho_loader(data_dir=data_dir, split='development',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=num_workers, return_reference=True, augment=hp.spec_augmentation, dataset_name=hp.dataset)

    validation_beam = get_clotho_loader(data_dir=data_dir, split='validation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=num_workers,
                                        shuffle=False,
                                        drop_last=False,
                                        return_reference=True,
                                        dataset_name=hp.dataset)

    evaluation_beam = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=num_workers,
                                        shuffle=False,
                                        drop_last=False,
                                        return_reference=True,
                                        dataset_name=hp.dataset)
    test_data = get_test_data_loader(data_dir=test_data_dir,
                                     batch_size=32,
                                     nb_t_steps_pad='max',
                                     num_workers=num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     input_pad_at='start',
                                     )
    logging.info(str(model))

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(training_data)))

    logging.info('Total Model parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # epoch = 1
    print('mixup is ',hp.mixup)
    if hp.mixup:
        mixup_augmentation = Mixup(mixup_alpha=1.0)
    else :
        mixup_augmentation = None
    # pdb.set_trace()
    if hp.mode == 'train':

        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            
            # pdb.set_trace()
            tr_loss = train(epoch, hp.training_epochs, hp.mixup, mixup_augmentation)
            logging.info('| epoch {:3d} | loss-mean-text {:5.4f}'.format(
                    epoch, tr_loss))
            scheduler.step(epoch)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }, '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            
            
            if epoch % 5== 0 or epoch > 10 or args.finetune_ce:
                eval_all(evaluation_beam, eos_ind=eos_ind, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
                # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                #             beam_size=2, eval_model=eval_model)
                eval_with_beam(evaluation_beam, max_len=30, sos_ind=sos_ind, eos_ind=eos_ind, word_dict_pickle_path=word_dict_pickle_path,
                            beam_size=3, eval_model=eval_model)
                eval_with_beam(evaluation_beam, max_len=30, sos_ind=sos_ind, eos_ind=eos_ind, word_dict_pickle_path=word_dict_pickle_path,
                            beam_size=4, eval_model=eval_model)

            else:
                eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)

            epoch += 1
            model.warm_up = False

    if hp.mode == 'eval':
        epoch = 21
       

        # Evaluation model score
        logging.info("The epoch is {}".format(epoch))
        # model.load_state_dict(torch.load("./models_trans/baseline_nowarmup_trans_tag_top5_losstag/{}.pt".format(epoch),map_location='cpu')['model'])
        # logging.info(" evaluation ")
        # eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path, eval_model=eval_model)
        # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
        #         beam_size=2, eval_model=eval_model)
        # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
        #             beam_size=3, eval_model=eval_model)

        eval_with_beam_visualization(evaluation_beam, max_len=30, sos_ind=sos_ind, eos_ind=eos_ind, word_dict_pickle_path=word_dict_pickle_path,
                        beam_size=4, eval_model=eval_model)
        
        # eval_with_beam_csv(evaluation_beam, max_len=30, sos_ind=sos_ind, eos_ind=eos_ind, word_dict_pickle_path=word_dict_pickle_path,
        #                 beam_size=4, eval_model=eval_model)
        
        # print("epoch is {}".format(epoch))
        # sys.exit()
        # epoch += 1

    elif hp.mode == 'test':
        model.load_state_dict(torch.load("./models/seed1234_truth_add_pretraintagloss_rl/55.pt",map_location='cpu')['model'])
        test_with_beam(test_data, beam_size=4, eval_model=eval_model,name="seed1234_truth_add_pretraintagloss_rl_analysis")

