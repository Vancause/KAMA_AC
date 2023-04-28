import os
import argparse
import time
lr = 5e-4
encoder_lr = 5e-5
training_epochs = 20
# name = 'test_one'
name = 'models_newtrans/test'#ResNet38_NewTF_tagloss_keywords_1e_7
cuda = 'cuda:0' # 
bs = 32
mode = 'train'
threshold_t = 0.4
topk_keywords = 3
resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# load_pretrain_emb
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
          f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
          ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/final_topk4'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 7
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/final_topk7'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 7
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/final_topk9'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 9
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
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
# name = 'models_audiocaps/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:4'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer/17.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} --resume {resume}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:4'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/lstm_new/17.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} --resume {resume}'
#           ) # --load_pretrain_emb 

# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/test'
# cuda = 'cuda:5'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# resume = 'models_audiocaps/lstm_new/17.pt'
# mode = 'eval'
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords} --dataset {dataset} --resume {resume}'
#            )






# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk8_newresidual_newnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 8
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk9_newresidual_newnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 9
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk10_newresidual_newnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 10
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk11_newresidual_newnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 11
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 20
# # name = 'test_one'
# name = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk13_newresidual_newnorm'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 13
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords}'
#           ) # --load_pretrain_emb 
