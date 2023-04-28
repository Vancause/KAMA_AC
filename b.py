import os
import argparse
import time
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_final_abalation/keywords3'
# cuda = 'cuda:2'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 3
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
#            )

lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 25
# name = 'test_one'
name = 'LSTM_final_abalation/keywords3_'
cuda = 'cuda:4'
bs = 32
decoder = 'AttDecoder'
topk_keywords = 3
# load_pretrain_emb
resume ='LSTM_final_abalation/keywords7/5.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
           )


lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 25
# name = 'test_one'
name = 'LSTM_final_abalation/keywords4_'
cuda = 'cuda:4'
bs = 32
decoder = 'AttDecoder'
topk_keywords = 4
# load_pretrain_emb
resume ='LSTM_final_abalation/keywords7/5.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
           )

lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 25
# name = 'test_one'
name = 'LSTM_final_abalation/keywords7_'
cuda = 'cuda:4'
bs = 32
decoder = 'AttDecoder'
topk_keywords = 7
# load_pretrain_emb
# resume ='models_trans/lstm_final/15.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
           )


lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 25
# name = 'test_one'
name = 'LSTM_final_abalation/keywords9_'
cuda = 'cuda:4'
bs = 32
decoder = 'AttDecoder'
topk_keywords = 9
# load_pretrain_emb
# resume ='models_trans/lstm_final/15.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
           )


# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_final_abalation/keywords4'
# cuda = 'cuda:2'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 4
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords}'
#            )
