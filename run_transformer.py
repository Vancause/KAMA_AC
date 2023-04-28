import os
import argparse
import time
print("action")


### method 
# one layer transformer
# lr = 1e-7
# training_epochs = 25
# # name = 'test_one'
# name = 'ResNet38_TF_le_7'
# cuda = 'cuda:0'
# bs = 32

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --load_pretrain_emb ')


# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss'
# cuda = 'cuda:0'
# bs = 32

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --load_pretrain_emb --use_tags_loss')

### find training strategy 
# lr = 1e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_keywords_1e_7'
# cuda = 'cuda:7'
# bs = 32

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --use_tags_t') # --load_pretrain_emb 

#  cuda 7
# lr = 3e-4   
# encoder_lr = 1e-4
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_1e_3_cuda7'
# cuda = 'cuda:7'
# bs = 32
# resume = 'models_trans/ResNet38_TF_tagloss_1e_3_cuda7/2.pt'

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb --encoder_lr {encoder_lr}') # --load_pretrain_emb 


# cuda 5
# lr = 3e-4   
# encoder_lr = 1e-4
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_keywords1038' # ResNet38_TF_tagloss_keywords1038   new_test
# cuda = 'cuda:5'
# bs = 32
# resume = 'models_trans/ResNet38_TF_tagloss_1e_3_cuda7/2.pt'

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb --encoder_lr {encoder_lr}') # --load_pretrain_emb 

# cuda 7 
lr = 5e-4   
encoder_lr = 5e-5
training_epochs = 30
# name = 'test_one'
name = 'ResNet38_TF_tagloss_keywords500_cuda0_twolayer_0_1ratio' # ResNet38_TF_tagloss_keywords1038   new_test
cuda = 'cuda:0'
bs = 32
resume = 'models_trans/ResNet38_TF_tagloss_keywords500_ls_cuda5_twolayer/21.pt'
mode = 'train'
# load_pretrain_emb
os.system(f' python train.py --lr {lr}'
          f' --training_epochs {training_epochs} '
          f'--name {name}  --spec_augmentation --label_smoothing '
          f'--batch_size {bs} --device {cuda} --use_tags_loss --encoder_lr {encoder_lr} --mode {mode} ') # --load_pretrain_emb 

# ###  
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_keywords1038_ls_cuda4_onelayer' # ResNet38_TF_tagloss_keywords1038   new_test
# cuda = 'cuda:3'
# bs = 32
# resume = 'models_trans/ResNet38_TF_tagloss_1e_3_cuda7/2.pt'

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb --encoder_lr {encoder_lr}') # --load_pretrain_emb 

### cuda 5 , 0
# lr = 5e-4   
# encoder_lr = 5e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_keywords500_cuda0_twolayer' # ResNet38_TF_tagloss_keywords1038   new_test # keywords extracted from train set   no label smooth
# cuda = 'cuda:0'
# bs = 32
# resume = 'models_trans/ResNet38_TF_tagloss_keywords500_cuda0_twolayer/5.pt'

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --encoder_lr {encoder_lr} --resume {resume}') # --load_pretrain_emb 




# cuda 0
# lr = 5e-4   
# encoder_lr = 1e-4
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_keywords1038_ls_addtag' # ResNet38_TF_tagloss_keywords1038   new_test
# cuda = 'cuda:0'
# bs = 32
# resume = 'models_trans/ResNet38_TF_tagloss_1e_3_cuda7/2.pt'

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb --encoder_lr {encoder_lr} --use_tags_t') # --load_pretrain_emb 


#cuda 0
# lr = 1e-5
# training_epochs = 30
# # name = 'test_one'
# name = 'ResNet38_TF_tagloss_1e_3_ratia_cuda0'   # 0.98
# cuda = 'cuda:0'
# bs = 32

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb') # --load_pretrain_emb 

# cuda 4
# lr = 1e-5   
# training_epochs = 30
# # name = 'test_one'
# name = 'test'
# cuda = 'cuda:4'
# bs = 32

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --use_tags_loss --load_pretrain_emb')  # --load_pretrain_emb 




### new _trans ,audio embs, keyword embs, tag loss
# lr = 3e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'New_transformer_use_tagemb'
# cuda = 'cuda:7'
# bs = 32
# resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb'
        #    )

### tans, audio embs, 
# lr = 3e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'transformer'
# cuda = 'cuda:6'
# bs = 32
# # resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb')

# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'New_transformer_use_tagemb_addforward_lr_oritrans'
# cuda = 'cuda:5'
# bs = 32
# # resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb')

# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'New_transformer_use_tagemb_addforward_lr_gated_oritrans'
# cuda = 'cuda:7'
# bs = 32
# # resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb')

# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'New_transformer_use_tagemb_addforward_lr_gated'
# cuda = 'cuda:4'
# bs = 32
# # resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb')




# time.sleep(3600*3.5)
### baseline 
# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'baseline'
# cuda = 'cuda:7'
# bs = 16
# resume = 'models_trans/test_text_embeddings_addinto_trans/19.pt' 

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb'
#            )


# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'eval'
# cuda = 'cuda:0'
# bs = 32
# resume = 'models_trans/baseline_true/15.pt' 
# # load_pretrain_emb
# mode = 'eval'
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb --mode {mode}'
#            )

## baseline_true_newtagmodel      tag false
# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'baseline_true_newtagmodel_top5_nolosstag'
# cuda = 'cuda:7'
# bs = 16
# resume = 'models_trans/baseline_true/15.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb '
#            )

# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'baseline_true_newtagmodel_threshold_newtrans'
# cuda = 'cuda:3'
# bs = 32
# resume = 'models_trans/baseline_true_newtagmodel_threshold/5.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb'
#            )

### baseline_true_newtagmodel project
# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'baseline_true_newtagmodel_proj'
# cuda = 'cuda:3'
# bs = 16
# resume = 'models_trans/baseline_true/15.pt' 
# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb '
#            )

#### baseline+fasttext+tag
# lr = 5e-4 
# training_epochs = 40
# # name = 'test_one'
# name = 'baseli4e+sfasttext+tag'
# cuda = 'cuda:2'
# bs = 16
# resume = 'models_trans/test_text_embeddings_addinto_trans/19.pt' 

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb'
#            )

#### 
# lr = 5e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'basline+fasttext+tag_freezecnn_truth'
# cuda = 'cuda:7'
# bs = 32
# resume = 'models_trans/test_text_embeddings_addinto_trans/19.pt' 

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb --freeze_cnn'
#            )

### finetune 
# lr = 1e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'test'
# cuda = 'cuda:2'
# bs = 32
# resume = 'models_trans/basline+fasttext+tag_freezecnn_truth/20.pt' 

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb --resume {resume} --finetune_ce'
#            )

# lr = 1e-4
# training_epochs = 40
# # name = 'test_one'
# name = 'test'
# cuda = 'cuda:0'
# bs = 32
# resume = 'models_trans/test/4.pt' 

# # load_pretrain_emb
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --load_pretrain_cnn --device {cuda} --load_pretrain_emb --resume {resume}'
#            )



### lstm 


# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'lstm_base'
# cuda = 'cuda:4'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# resume ='models_trans/lstm_base/5.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder}'
#            )

# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_tag_prev_tagloss_endt2end'
# cuda = 'cuda:4'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --preword_emb --use_tags_loss'
#            )


# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_tagloss_endt2end'
# cuda = 'cuda:4'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--use_tags_loss'
#            )


# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_prev_tagloss_endt2end'
# cuda = 'cuda:4'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--preword_emb --use_tags_loss'
#            )

# lr = 5e-4
# training_epochs = 25
# # name = 'test_one'
# name = 'LSTM_tag_tagloss_endt2end'
# cuda = 'cuda:4'
# bs = 32
# decoder = 'AttDecoder'
# # load_pretrain_emb
# # resume ='models_trans/lstm_final/15.pt' 
# os.system(f' python train.py --lr {lr}'
#           f' --training_epochs {training_epochs} '
#           f'--name {name}  --spec_augmentation --label_smoothing '
#           f'--batch_size {bs} --device {cuda} --decoder {decoder} '
#           f'--tag_emb --use_tags_loss'
#            )



