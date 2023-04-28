import os
import argparse
import time
print("action")

### ------ featurecut transformer  clotho
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# kl_training_epochs = 30
# # name = 'test_one'
# name = 'featurecut_final/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# decoder = 'Transformer'
# mode = 'eval'
# # threshold_t = 0.4
# topk_keywords = 5
# sample_ratio = 0.8
# aug_type = 'feature'
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'featurecut_final/transformer_featurecut30_audiocapsreal_attnweights2_epoch30_new/22.pt'

# # resume = 'featurecut_final/transformer_featurecut30_audiocaps/23.pt'

# # load_pretrain_emb
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--use_tags_t   --use_newtrans --topk_keywords {topk_keywords} --decoder {decoder} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} --resume {resume}'
#           ) # --load_pretrain_emb 

# # ### ------ featurecut attmodel  clotho
# lr = 5e-4   
# encoder_lr = 1e-4
# training_epochs = 30
# kl_training_epochs = 30
# # name = 'test_one'
# name = 'featurecut_final/attmodel_clotho_featurecut_new1s'
# cuda = 'cuda:5'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.8
# aug_type = 'feature'
# # load_pretrain_emb
# resume = 'featurecut_final/attmodel_clotho_featurecut/10.pt'
# # resume = 'models_featurecut/featurecut_final/attmodel_featurecut/20.pt'
# dataset = 'Clotho'
# mode = 'train'
# # resume ='models_featurecut/featurecut_final/attmodel_featurecut/1.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --mode {mode} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} '
#            )

lr = 5e-4   
encoder_lr = 5e-4
training_epochs = 30
kl_training_epochs = 30
# name = 'test_one'
name = 'featurecut_final/attmodel_clotho_base'
cuda = 'cuda:5'
bs = 32
decoder = 'AttDecoder'
sample_ratio = 0.8
aug_type = 'none'
# load_pretrain_emb
resume = 'featurecut_final/attmodel_clotho_base/8.pt'
# resume = 'models_featurecut/featurecut_final/attmodel_featurecut/20.pt'
dataset = 'Clotho'
mode = 'train'
# resume ='models_featurecut/featurecut_final/attmodel_featurecut/1.pt' 
os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --decoder {decoder} '
          f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} --mode {mode} '
          f'--aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} --resume {resume}'
           )

### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ 

# ### ------ featurecut lstm  AudioCaps

# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# kl_training_epochs = 10
# aug_type = 'feature'
# sample_ratio = 0.8
# # name = 'test_one'
# name = 'featurecut_final/lstm_base_featurecut'
# cuda = 'cuda:7'
# bs = 32
# decoder = 'AttDecoder'
# topk_keywords = 5
# dataset = 'AudioCaps' # AudioCaps Clotho
# mode = 'train'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
#           f'--encoder_lr {encoder_lr} --dataset {dataset} --freeze_cnn --load_pretrain_cnn '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --kl_training_epochs {kl_training_epochs}'
#            )

### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ 

### ------ featurecut lstm  AudioCaps
# lr = 5e-4
# encoder_lr = 1e-4
# training_epochs = 30
# kl_training_epochs = 10
# # name = 'test_one'
# name = 'featurecut_final/test'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:0'
# bs = 32
# decoder = 'Transformer'
# mode = 'eval'
# threshold_t = 0.4
# topk_keywords = 5
# sample_ratio = 0.8
# nhead_t = 4
# nhid_t = 128
# nlayers_t = 2 
# dim_feedforward = 2048
# aug_type = 'feature'
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'featurecut_final/cnn10_transformer_audiocaps_base_2048_new/30.pt'
# # load_pretrain_emb
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --encoder_lr {encoder_lr} --mode {mode} '
#           f'--decoder {decoder} --use_mei '
#           f'--dataset {dataset} '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} --resume {resume}'
#           ) # --load_pretrain_emb 



# lr = 5e-4
# encoder_lr = 1e-4
# training_epochs = 30
# kl_training_epochs = 30
# # name = 'test_one'
# name = 'featurecut_final/cnn10_transformer_audiocaps_featurecut_2048_allattends_new1'#ResNet38_NewTF_tagloss_keywords_1e_7
# cuda = 'cuda:5'
# bs = 32
# decoder = 'Transformer'
# mode = 'train'
# threshold_t = 0.4
# topk_keywords = 5
# sample_ratio = 0.8
# nhead_t = 4
# nhid_t = 128
# nlayers_t = 2 
# dim_feedforward = 2048

# aug_type = 'feature'
# dataset = 'AudioCaps' # AudioCaps Clotho
# resume = 'featurecut_final/cnn10_transformer_audiocaps_featurecut_2048_allattends_new/15.pt'
# # load_pretrain_emb
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda}  --encoder_lr {encoder_lr} --mode {mode} '
#           f'--decoder {decoder} --use_mei '
#           f'--nhead_t {nhead_t} --nhid_t {nhid_t} --nlayers_t {nlayers_t} --dim_feedforward {dim_feedforward} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset} --kl_training_epochs {kl_training_epochs} --resume {resume}'
#           ) # --load_pretrain_emb 







### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ ### ------ 


# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# kl_training_epochs = 20
# # name = 'test_one'
# name = 'featurecut_final/attmodel_featurecut30_epoch10'
# cuda = 'cuda:2'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.8
# aug_type = 'feature'
# # load_pretrain_emb
# resume = 'featurecut_final/attmodel_featurecut30/15.pt'
# # resume = 'models_featurecut/featurecut_final/attmodel_featurecut/20.pt'
# dataset = 'Clotho'
# mode = 'train'

# # resume ='models_featurecut/featurecut_final/attmodel_featurecut/1.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} --mode {mode} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset}  --kl_training_epochs {kl_training_epochs}'
#            )


# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'featurecut_final/attmodel_random'
# cuda = 'cuda:5'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.8
# aug_type = 'random'
# # load_pretrain_emb
# dataset = 'Clotho'
# resume ='models_featurecut/featurecut_final/attmodel_featurecut/1.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --dataset {dataset}'
#            )


#--------------

# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.8
# aug_type = 'feature'
# name = 'ye_featurecut_ratio0_8_freezecnn'
# # name = 'eval'
# cuda = 'cuda:0'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn --load_pretrain_cnn'
#            )

# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.8
# aug_type = 'feature'
# name = 'ye_featurecut_base'
# # name = 'eval'
# cuda = 'cuda:0'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--freeze_cnn --load_pretrain_cnn'
#            )

# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 40
# # name = 'test_one'
# name = 'featurecut_final/attmodel_featurecut'
# cuda = 'cuda:7'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.7
# aug_type = 'feature'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut '
#            )


####### attmodel klloss
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 40
# # name = 'test_one'
# name = 'featurecut_final/attmodel_featurecut'
# cuda = 'cuda:7'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.7
# aug_type = 'feature'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut '
#            )

# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 40
# # name = 'test_one'
# name = 'featurecut_final/attmodel_random'
# cuda = 'cuda:7'
# bs = 32
# decoder = 'AttDecoder'
# sample_ratio = 0.7
# aug_type = 'random'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut '
#            )



# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 40
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7_noaugloss_load_pretrained_cnn'
# # name = 'eval'
# cuda = 'cuda:5'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --load_pretrain_cnn --freeze_cnn'
#            )


#### -----------------

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.9
# aug_type = 'feature'
# name = 'featurecut_ratio0_9'
# # name = 'eval'
# cuda = 'cuda:6'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.8
# aug_type = 'feature'
# name = 'featurecut_ratio0_8'
# # name = 'eval'
# cuda = 'cuda:6'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7'
# # name = 'eval'
# cuda = 'cuda:6'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.6
# aug_type = 'feature'
# name = 'featurecut_ratio0_6'
# # name = 'eval'
# cuda = 'cuda:6'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.9
# aug_type = 'feature'
# name = 'featurecut_ratio0_9_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.8
# aug_type = 'feature'
# name = 'featurecut_ratio0_8_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.6
# aug_type = 'feature'
# name = 'featurecut_ratio0_6_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.5
# aug_type = 'feature'
# name = 'featurecut_ratio0_5_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )



# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 40
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7_noaugloss_load_pretrained_cnn'
# # name = 'eval'
# cuda = 'cuda:5'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --load_pretrain_cnn --freeze_cnn'
#            )

# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 40
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7_noaugloss_load_pretrained_cnn'
# # name = 'eval'
# cuda = 'cuda:5'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--load_pretrain_cnn --freeze_cnn'
#            )

# lr = 3e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.7
# aug_type = 'random'
# name = 'randomcut_ratio0_7_noaugloss_load_pretrained_cnn'
# # name = 'eval'
# cuda = 'cuda:5'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --load_pretrain_cnn --freeze_cnn'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.7
# aug_type = 'feature'
# name = 'featurecut_ratio0_7_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.6
# aug_type = 'feature'
# name = 'featurecut_ratio0_6_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.5
# aug_type = 'feature'
# name = 'featurecut_ratio0_5_noaugloss'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut'
#            )
