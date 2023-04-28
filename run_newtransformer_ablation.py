import os
import argparse
import time
print("action")
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_baseline'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda}  --encoder_lr {encoder_lr} --mode {mode} '
#         #   f'--use_newtrans '
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_tagloss'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 6
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#         #   f'--use_newtrans'
#           ) # --load_pretrain_emb 
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '512_lstm_tagloss_new'
# cuda = 'cuda:7'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--use_tags_loss --encoder_lr {encoder_lr} '
#            )


lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 30
# name = 'test_one'
name = '512_lstm_tagloss_prev_tag_new'
cuda = 'cuda:0'
bs = 32
decoder = 'AttDecoder'
# load_pretrain_emb
# resume ='models_trans/lstm_final/15.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
           )

lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 30
# name = 'test_one'
name = '512_lstm_tagloss_prev_new'
cuda = 'cuda:0'
bs = 32
decoder = 'AttDecoder'
# load_pretrain_emb
# resume ='models_trans/lstm_final/15.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
           )


# final 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk5'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk5_newresidual_addnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_toptags1_and_threshold'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# resume = 'models_newtrans/R38_newtrans_tagloss_toptags1_and_threshold/11.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t  --use_threshold_t --threshold_t {threshold_t}  --use_newtrans --resume {resume}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk5'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk6'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 6
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk7'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 7
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk8'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 8
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk9'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 9
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = '256_2048_models_newtrans/R38_newtrans_tagloss_toptags1_and_topk10'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 10
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 




# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_toptags1_and_threshold0_35_masks'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.35
# resume = 'models_newtrans/R38_newtrans_tagloss_toptags1_and_threshold/11.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t  --use_threshold_t --threshold_t {threshold_t}  --use_newtrans '
#           ) # --load_pretrain_emb 


# # use_tags_t = False
# #     use_tags_loss = False
# #     use_threshold_t = False 
# #     threshold_t = 0.4
# #     use_newtrans = False