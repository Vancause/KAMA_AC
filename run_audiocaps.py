import os
import argparse
import time


#### --------
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/lstm_base'
# cuda = 'cuda:0'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# mode = 'train'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
#           f'--encoder_lr {encoder_lr} --dataset {dataset} --freeze_cnn --load_pretrain_cnn'
#            )


#### ----------
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_newdataset_refine2_3epochdeclay'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:1'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_newdataset/12.pt'
# nhead_t = 4
# nhid_t = 256
# nlayers_t = 2 
# dim_feedforward = 1024
# optimizer_epoch = 4
# optimizer_delay = 0.5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--optimizer_epoch {optimizer_epoch} --optimizer_delay {optimizer_delay} '
#           ) # --load_pretrain_emb 

# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_newdataset_refine2_3epochdeclay_new'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:1'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_newdataset_refine2_3epochdeclay_new/12.pt'
# nhead_t = 4
# nhid_t = 256
# nlayers_t = 2 
# dim_feedforward = 1024
# optimizer_epoch = 5
# optimizer_delay = 0.5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--optimizer_epoch {optimizer_epoch} --optimizer_delay {optimizer_delay} --resume {resume}'
#           ) # --load_pretrain_emb 

lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 30
# name = 'test_one'
name = 'models_audiocaps/lstm_newdataset_refine2_refineyici'
cuda = 'cuda:7'
bs = 32
decoder = 'AttDecoder'
topk_keywords = 5
dataset = 'AudioCaps' # AudioCaps Clotho
mode = 'train'
# load_pretrain_emb
# resume ='models_trans/lstm_final/15.pt' 
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords} --dataset {dataset}'
           )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/test'#ResNet38_NewTF_tagloss_keywords_1e_7 # test1 tes
# cuda = 'cuda:0'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_newdataset_refine2/16.pt'
# nhead_t = 4
# nhid_t = 256
# nlayers_t = 2 
# dim_feedforward = 1024
# optimizer_epoch = 5
# optimizer_delay = 0.5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--optimizer_epoch {optimizer_epoch} --optimizer_delay {optimizer_delay} --resume {resume}'
#           ) # --load_pretrain_emb 



# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/test'
# cuda = 'cuda:3'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# mode = 'eval'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords} --dataset {dataset}'
#            )


#3.23  # cuda 5
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_bs32_spatialdropout'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_/5.pt'
# nhead_t = 4
# nhid_t = 256
# nlayers_t = 2 
# dim_feedforward = 1024
# optimizer_epoch = 5
# optimizer_delay = 0.5
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--optimizer_epoch {optimizer_epoch} --optimizer_delay {optimizer_delay} '
#           ) # --load_pretrain_emb 


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_bs32_new'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_/5.pt'
# nhead_t = 4
# nhid_t = 256
# nlayers_t = 2 
# dim_feedforward = 1024
# optimizer_epoch = 10
# optimizer_delay = 0.1
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--optimizer_epoch {optimizer_epoch} --optimizer_delay {optimizer_delay} '
#           ) # --load_pretrain_emb 



# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_bs32_newlayer1'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_3_layer/5.pt'
# nhead_t = 8
# nhid_t = 512
# nlayers_t = 1
# dim_feedforward = 2048
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward}'
#           ) # --load_pretrain_emb 



# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_bs64'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 64
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_3_layer/5.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset}'
#           ) # --load_pretrain_emb 



###---------------------------------------------------------------------------------------------------------------------------------------------

# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/lstm_new_2'
# cuda = 'cuda:3'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords} --dataset {dataset}'
#            )


###---------------------------------------------------------------------------------------------------------------------------------------------

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer_2_layer_top7'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:7'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 7
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_3_layer/5.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset}'
#           ) # --load_pretrain_emb 

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_audiocaps/transformer_new/21.pt'
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
# name = 'models_audiocaps/lstm_new'
# cuda = 'cuda:5'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --topk_keywords {topk_keywords} --dataset {dataset}'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'models_audiocaps/transformer'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'models_newtrans/R38_newtrans_tagloss_topktags_and_topk6_newresidual_newnorm/16.pt'
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --dataset {dataset}'
#           ) # --load_pretrain_emb 



