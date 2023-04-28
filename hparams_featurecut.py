from pathlib import Path


class hparams:
    batch_size = 32
    nhid = 256
    output_dim_encoder = 512   # 256   512
    output_dim_h_decoder = 512  # 256  512
    att_size = 512    # 256   512
    emb_size = 512    # 256 512
    dropout_p_encoder = 0.25
    dropout_p_decoder = 0.25
    max_out_t_steps = 30

    nhead = 4
    natt = 512
    nlayers = 2
    ninp = 64
    ntoken = 4367 + 1
    # ntoken = 4371 + 1
    clip_grad = 2.5
    lr = 3e-4  # learning rate 3e-4
    beam_width = 3
    training_epochs = 50
    log_interval = 100
    checkpoint_save_interval = 5
    decoder = 'AttDecoder'
    # decoder = 'Transformer'

    seed = 2023
    device = 'cuda:7'   #'cuda:0' 'cuda:1' 'cpu'
    mode = 'train'
    name = 'base'
    nkeyword = 4979
    # augmentation mixup
    mixup = False
    label_smoothing = True
    load_pretrain_cnn = True
    # freeze_cnn = True
    freeze_classifer = True
    load_pretrain_emb = False
    load_pretrain_model = False
    spec_augmentation = True
    scheduler_decay = 0.98
    tmp_name = 'one_stage_featcut_cuda01'
    NopadLoss = True
    tag_emb = False  # 
    multiScale = False
    preword_emb = False  # 
    two_stage_cnn = False
    usingLM = False
    multi_gpu = False
    train_all = False # 
### Transformer setting 
    nhead_t = 4
    nhid_t = 256  # 192 300
    nlayers_t = 2
    ninp_t = 64
    dim_feedforward = 1024 # 2048 1200
    use_tags_t = False
    use_tags_loss = False
    use_threshold_t = False 
    threshold_t = 0.4
    use_newtrans = False
    topk_keywords = 5

# Train tagging model
    train_tag = False
    save_name = '300_one_3fc_truth'
    class_num = 500 # 300
    tag_mixup  = True   #
    tag_focalLoss = False
    tag_GMAP = False
    tag_specMix = False


# sample featurecut 
    sample_ratio = 0.9
    aug_type = 'feature' # random, none 
    encoder_lr = 5e-5
# data(default)
    data_dir = Path(r'./create_dataset/data/data_splits')
    eval_data_dir = r'./create_dataset/data/data_splits/evaluation'
    train_data_dir = r'./create_dataset/data/data_splits/development'
    # test_data_dir = r'./create_dataset/data/test_data'
    test_data_dir = r'./create_dataset/data/analysis_data'
    word_dict_pickle_path = r'./create_dataset/data/pickles/words_list.p'
    word_freq_pickle_path = r'./create_dataset/data/pickles/words_frequencies.p'
    word_freq_reciprocal_pickle_path = r'./create_dataset/data/pickles/words_weight.pickle'
    # pretrain_model
    # tag_keyword_pickle_path = r'./audio_tag/word_list_pretrain_rules.p'
    # tagging_to_embs = r'./audio_tag/TaggingToEmbs.p'
    # tag_keyword_pickle_path = r'./audio_tag/word_list_pretrain_rules_train_new.p'
    # tagging_to_embs = r'./audio_tag/TaggingToEmbs_train_new.p'
    tag_keyword_pickle_path = r'./audio_tag/word_list_pretrain_rules_train_new.p'
    # tagging_to_embs = r'./audio_tag/TaggingToEmbs_train_new.p'
    # tagging_to_embs = r'./audio_tag/TaggingToEmbs_allwords.p'
    tagging_to_embs = r'./audio_tag/TaggingToEmbs_500_train.p'


    pretrain_emb_path = './create_dataset/data/pickles/fasttext_300d.p'
    #pretrain_emb_path = r'./bert_last_hidden.pickle'
    # pretrain_cnn_path = r'./models/tag_models_baseline_finetune/TagModel_25.pt'
    # pretrain_cnn_path = r'./models/tag_models_baseline_finetune/TagModel_40.pt'
    # pretrain_cnn_path = r'./models/300_one_3fc_finetune/TagModel_40.pt'
    # pretrain_cnn_path = r'./models/300_one_3fc_truth_finetune/TagModel_40.pt'  # TagModel_40.pt'
    pretrain_cnn_path = r'./models/500classies_finetune/TagModel_40.pt'  # TagModel_40.pt'
    pretrain_cnn_path_audiocaps = r'./models/500classies_audiocaps_finetune/TagModel_25.pt'  # TagModel_40.pt'

    pretrain_model_path = r'models/baseline/30.pt'
    #eval dir
    eval_dir = "seed1111/"
    data_dir_audiocaps = Path(r'./create_dataset/AudioCaps/data_splits')
    # tagging_to_embs_audiocaps = r'./audio_tag/AudioCaps/TaggingToEmbs_500_train.p'
    # word_dict_pickle_path_audiocaps = r'./create_dataset/AudioCaps/pickles/words_list.p'
    tagging_to_embs_audiocaps = r'./create_dataset/AudioCaps/data_splits/TaggingToEmbs_500_train.p'
    word_dict_pickle_path_audiocaps = r'./create_dataset/AudioCaps/data_splits/words_list.p'
    # ntoken 5046


