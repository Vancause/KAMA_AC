import os
import argparse
import time
print("action")

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:3'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# resume = 'models_newtrans/R38_baseline/20.pt'
# # load_pretrain_emb
# os.system(f' python visualization.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t --topk_keywords {topk_keywords} --resume {resume}'
#           ) # --load_pretrain_emb 
#R38_baseline
### attmodel 
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# # resume = 'models_trans/lstm_base/17.pt'   
# resume = '256_lstm_tagloss_prev_tag/23.pt'
# decoder = 'AttDecoder'
# # load_pretrain_emb
# os.system(f' python visualization.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--tag_emb --preword_emb --use_tags_t --topk_keywords {topk_keywords} --resume {resume} --decoder {decoder}'
#           ) # --load_pretrain_emb 

# transformer
lr = 5e-4
encoder_lr = 5e-5
training_epochs = 20
# name = 'test_one'
name = 'models_newtrans/test'#ResNet38_NewTF_tagloss_keywords_1e_7
cuda = 'cuda:0'
bs = 32
mode = 'eval'
threshold_t = 0.4
topk_keywords = 3
dataset = 'Clotho'
resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk5_newresidual_newnorm/13.pt'
# load_pretrain_emb
os.system(f' python visualization.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
          f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --resume {resume} --dataset {dataset}'
          ) # --load_pretrain_emb 

