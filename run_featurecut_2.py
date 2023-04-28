import os
import argparse
import time
print("action")

# 

# base 
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# name = '512_baseline'
# # name = 'eval'
# cuda = 'cuda:2'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm/2.pt'
# aug_type = 'none'
# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} --aug_type {aug_type} --freeze_cnn'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.5
# aug_type = 'feature'
# name = 'featurecut_ratio0_5'
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
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )

### randomecut
# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.9
# aug_type = 'random'
# name = 'test'
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
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )

# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.8
# aug_type = 'random'
# name = 'randomcut_ratio0_8'
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
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )


# lr = 5e-4
# encoder_lr = 5e-5
# training_epochs = 30
# sample_ratio = 0.7
# aug_type = 'random'
# name = 'randomcut_ratio0_7'
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
#           f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
#            )



# lr = 5e-4
# encoder_lr = 1e-5
# training_epochs = 40
# name = 'baseline_cnn_lstm_again2'
# # name = 'eval'
# cuda = 'cuda:6'
# bs = 32
# # load_pretrain_emb
# # resume = 'models_featurecut/baseline_cnn_lstm/2.pt'
# mode = 'train'
# os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr}'
#            )

##### new

## randomecut
lr = 5e-4
encoder_lr = 5e-5
training_epochs = 30
sample_ratio = 0.9
aug_type = 'random'
name = 'randomcut_ratio0_9_noaugloss'
# name = 'eval'
cuda = 'cuda:6'
bs = 32
# load_pretrain_emb
# resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

mode = 'train'
os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
          f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
           )

lr = 5e-4
encoder_lr = 5e-5
training_epochs = 30
sample_ratio = 0.8
aug_type = 'random'
name = 'randomcut_ratio0_8_noaugloss'
# name = 'eval'
cuda = 'cuda:6'
bs = 32
# load_pretrain_emb
# resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

mode = 'train'
os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
          f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
           )


lr = 5e-4
encoder_lr = 5e-5
training_epochs = 30
sample_ratio = 0.7
aug_type = 'random'
name = 'randomcut_ratio0_7_noaugloss'
# name = 'eval'
cuda = 'cuda:6'
bs = 32
# load_pretrain_emb
# resume = 'models_featurecut/baseline_cnn_lstm_again/2.pt'

mode = 'train'
os.system(f' python train_original_featurecut_FAT_drop.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs}  --device {cuda} --mode {mode} --encoder_lr {encoder_lr} '
          f'--sample_ratio {sample_ratio} --aug_type {aug_type} --use_featurecut --freeze_cnn'
           )
