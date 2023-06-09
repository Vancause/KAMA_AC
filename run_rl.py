import os
import argparse

# augmentation+smoothing+pretrain_cnn+freeze_cnn
# lr = 5e-5
# # lr=4.26e-5
# training_epochs = 100
# # name = 'seed1234_rl_truth'  # no pre no tag
# name = 'final_sub_rl_notagpre'
# # name = 'test'
# scheduler_decay = 0.99
# caption_model_path ='./models/seed1234_truth_trainall_nopre_notag/30.pt'
# os.system(f'python train_rl.py --lr {lr} --scheduler_decay {scheduler_decay} '
#           f'--training_epochs {training_epochs} --name {name} '
#           f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
#           f'--spec_augmentation  --label_smoothing --resume models/final_sub_rl_notagpre/20.pt')

lr = 5e-5
# lr=4.26e-5
training_epochs = 100
# name = 'seed1234_rl_truth'  # no pre no tag
name = 'seed1234_truth_add_pretraintagloss_rl'
# name = 'test'
scheduler_decay = 0.99
caption_model_path ='./models/seed1234_truth_add_tagloss/30.pt'
os.system(f'python train_rl.py --lr {lr} --scheduler_decay {scheduler_decay} '
          f'--training_epochs {training_epochs} --name {name} '
          f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
          f'--spec_augmentation  --label_smoothing ')

# lr = 5e-5
# # lr=4.26e-5
# training_epochs = 100
# name = 'seed1234_rl_nopre_notag_truth'
# # name = 'test'
# scheduler_decay = 0.99
# caption_model_path ='./models/seed1234_nopre_notag_truth/30.pt'
# # caption_model_path ='./models/seed1234_rl_truth/15.pt'
# os.system(f'python train_rl.py --lr {lr} --scheduler_decay {scheduler_decay} '
#           f'--training_epochs {training_epochs} --name {name} '
#           f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
#           f'--spec_augmentation  --label_smoothing')

## 
# lr = 5e-5
# # lr=4.26e-5
# training_epochs = 100
# name = 'seed1234_truth_add_tagloss_rl'
# # name = 'test'
# scheduler_decay = 0.99
# caption_model_path ='./models/seed1234_truth_add_tagloss/30.pt'
# # caption_model_path ='./models/seed1234_rl_truth/15.pt'
# os.system(f'python train_rl.py --lr {lr} --scheduler_decay {scheduler_decay} '
#           f'--training_epochs {training_epochs} --name {name} '
#           f'--load_pretrain_model --pretrain_model_path {caption_model_path} '
#           f'--spec_augmentation  --label_smoothing --resume models/seed1234_truth_add_tagloss_rl/49.pt')