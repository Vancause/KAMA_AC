from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from numpy import ndarray, recarray
from torch.utils.data import Dataset
from numpy import load as np_load
import pickle
import torch
import numpy as np
import os
from torch import cat as pt_cat, zeros as pt_zeros, \
    ones as pt_ones, from_numpy, Tensor
from torch.utils.data.dataloader import DataLoader
import pdb

class TagDataset(Dataset):
    # development,evaluation

    def __init__(self, data_dir: Path,
                 split: AnyStr, class_num, dataset_name='Clotho'):
        super(TagDataset, self).__init__()
        if split == "all":
            print("train all ~~~~")
            the_dir: Path = data_dir.joinpath("development")
            self.examples: List[Path] = sorted(the_dir.iterdir())[::5]
            the_dir1: Path = data_dir.joinpath("validation")
            self.examples.extend(sorted(the_dir1.iterdir())[::5])
            the_dir2: Path = data_dir.joinpath("evaluation")
            self.examples.extend(sorted(the_dir2.iterdir())[::5])
            tags1 = pickle.load(open('./audio_tag/audioTagNum_{}_fin_nv_train_expeople.pickle'.format("development"), 'rb'))
            tags2 = pickle.load(open('./audio_tag/audioTagNum_{}_fin_nv_train_expeople.pickle'.format("validation"), 'rb'))
            tags3 = pickle.load(open('./audio_tag/audioTagNum_{}_fin_nv_train_expeople.pickle'.format("evaluation"), 'rb'))
            print(len(tags1),len(tags2),len(tags3))
            self.tags = tags1.copy()
            self.tags.update(tags2)
            self.tags.update(tags3)
        else:
            the_dir: Path = data_dir.joinpath(split)
            if split == 'development':
                self.examples: List[Path] = sorted(the_dir.iterdir())
            else:
                self.examples: List[Path] = sorted(the_dir.iterdir())[::5]
            # with open('./audio_tag/audioTagNum_{}_fin_nv_train_500.pickle'.format(split), 'rb') as f:
            #     self.tags = pickle.load(f)
            with open('./audio_tag/{}/audioTagNum_{}_fin_nv_train_500.pickle'.format(dataset_name,split), 'rb') as f: # 1308
                self.tags = pickle.load(f)

            # if split == 'development':
            #     # pdb.set_trace()
            #     the_dir1: Path = data_dir.joinpath('validation')
            #     self.examples1: List[Path] = sorted(the_dir1.iterdir())[::5]
            #     self.examples = self.examples + self.examples1
            #     with open('./audio_tag/audioTagNum_{}_fin_nv.pickle'.format('validation'), 'rb') as f:
            #         self.tags1 = pickle.load(f)
            #     self.tags.update(self.tags1)
        self.change_split = {'development':'train','validation':'val','evaluation':'test'}
        self.dataset = dataset_name
        print("train {} examples and {} tags ".format(len(self.examples),len(self.tags)))

    def __len__(self) \
            -> int:
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray]:
        ex: Union[Path, recarray] = self.examples[item]
        ex: recarray = np_load(str(ex), allow_pickle=True)
        features = ex['features'].item()
        file_name = ex['file_name'].item()
        
        split = str(self.examples[item]).split('/')[3]
        if self.dataset == 'AudioCaps':
            # in_e = in_e[0]
            split = self.change_split[split]
        tag = self.tags[(split,file_name)]
        all_filenames = str(ex['file_name'].item())
        # return features, tag, all_filenames
        return features, tag, all_filenames
def load_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


# class TagDataset(Dataset):
#     # development,evaluation

#     def __init__(self, data_dir: Path,
#                  split: AnyStr, class_num):
#         super(TagDataset, self).__init__()
       
#         the_dir: Path = data_dir.joinpath(split)
#         self.examples: List[Path] = sorted(the_dir.iterdir())[::5]
#         # pdb.set_trace()
#         if split == 'development':
#             split = 'train'
#         elif split == 'validation':
#             split = 'val'
#         else:
#             split = 'test'
#         self.target_dict = load_pickle_file(f'/data/yzj_data/workspace/DCASE_task6_tag/data/Clotho/pickles/456/{split}_keywords_dict_456.p')
#         self.vocab = load_pickle_file(f'/data/yzj_data/workspace/DCASE_task6_tag/data/Clotho/pickles/456/tag_vocab_456.p')

#         print("train {} examples and {} tags ".format(len(self.examples),len(self.target_dict)))

#     def __len__(self) \
#             -> int:
#         return len(self.examples)

#     def __getitem__(self,
#                     item: int) \
#             -> Tuple[ndarray, ndarray]:
#         ex: Union[Path, recarray] = self.examples[item]
#         ex: recarray = np_load(str(ex), allow_pickle=True)
#         features = ex['features'].item()
#         file_name = ex['file_name'].item()
#         # pdb.set_trace()
#         split = str(self.examples[item]).split('/')[3]
#         kws = self.target_dict[file_name]
#         all_filenames = ex['file_name']
#         target = np.zeros(len(self.vocab))
#         if kws != []:
#             kws_index = np.array([self.vocab.index(kw) for kw in kws])
#             target[kws_index] = 1.
        
#         # return features, tag, all_filenames
#         return features, target
    

def collate_fn(batch, input_pad_at='start'):
    in_t_steps = max([i[0].shape[0] for i in batch])

    in_dim = batch[0][0].shape[-1]

    input_tensor, output_tensor = [], []
    # all_files = []
    all_nouse = []
    
    for i, (in_b, out_b, _) in enumerate(batch):
        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()

        input_tensor.append(tmp_in.unsqueeze_(0))
        tmp_out: Tensor = torch.Tensor(out_b)
        output_tensor.append(tmp_out.unsqueeze_(0))
        # all_files.extend(files)
    
    input_tensor = pt_cat(input_tensor)
    output_tensor = pt_cat(output_tensor)
    file_names = [i[2] for i in batch]
    # pdb.set_trace()
    # return input_tensor, output_tensor,all_files
    return input_tensor, output_tensor


def tag_loader(data_dir, split, batch_size, class_num=300, shuffle=True, drop_last=True,dataset_name='Clotho'):
    dataset = TagDataset(data_dir, split, class_num,dataset_name)
    if split ==  "development" or split == "all":
        drop_last = True
    else:
        drop_last = False
    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, num_workers=4)


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(r'../create_dataset/data/data_splits')
    test_data = tag_loader(data_dir=data_dir, split='evaluation',
                           batch_size=16, class_num=300)
    pred_output = []
    all_tag = []
    print(len(test_data))
    for i, (feature, tag, nouselist) in enumerate(test_data):
        tag_ = tag.cpu().numpy()
        if len(nouselist) != 0:
            tag_ = np.delete(tag_, nouselist, 0)
        # print(tag_.shape)
        all_tag.extend(tag_)
        # print(tag_.shape)

    all_tag = np.array(all_tag)
    for v in all_tag:
        b = np.argwhere(v == 1)
        print(len(b))
