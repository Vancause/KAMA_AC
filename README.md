# The code of master's thesis

This is the code of master's thesis which contains my previous two work ([MAAC](http://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Ye_19.pdf) and [FeatureCut](https://ieeexplore.ieee.org/document/9980325?denied=)). 

### Setting up the Code and Environment
1. Clone this repository: `https://github.com/Vancause/KAMA_AC.git`
2. Install pytorch >=1.8.0
3. Use pip to install dependencies: `pip install -r requirements.txt`
### Preparing the data
#### Clotho 
+ [Download](http://dcase.community/challenge2020/task-automatic-audio-captioning#download) the Clotho dataset for DCASE2021 Automated Audio Captioning challenge. And how to prepare training data and setup coco caption, please refer to [Dcase2020 BUPT team's](https://github.com/lukewys/dcase_2020_T6)
+ Enter the **audio_tag** directory. 
+ Firstly, run `python generate_word_list.py` to create words list `word_list_pretrain_rules.p` and tagging words to indexes of embedding layer `TaggingtoEmbs`. 
+ Then run `python generate_tag.py` to generate audioTagName\_{development/validation/evaluation}\_fin\_nv.pickle and audioTagNum\_{development/validation/evaluation}\_fin\_nv.pickle 

#### AudioCaps
+ Download the dataset from https://github.com/XinhaoMei/ACT.
+ Generate the `.npy` files through running `generate_audiocaps_files.py`

### Configuration
The training configuration is saved in the `hparams.py` and you can reset it to your own parameters.    
### Train the KAMA-AC model.
+ Run `python run_newtransformer.py` to train the KAMA-AC-T model.
+ Run `python run_lstm.py` to train the KAMA-AC-L model.
+ In the files `run_lstm.py or run_newtransformer.py`, you can modify hyper-parameters directly to run the ablations.
### Train the KAMA-AC model with FeatureCut.
+ Run `python run_featurecut.py` to train the KAMA-AC-L model.
+ In the files `run_lstm.py or run_newtransformer.py`, you can modify hyper-parameters directly to run the ablations

