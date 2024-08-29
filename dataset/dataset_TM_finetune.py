import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import random
import math

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name_tune=None, tokenizer_name=None, up_low_sep=False,multi_sep=False):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.up_low_sep = up_low_sep
        self.multi_sep = multi_sep
        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1 # [TODO] I think 513 (codebook_size+1) can be what ever, it will be croped out
        if dataset_name == 't2m':
            self.data_root = '/extra/xielab0/araujog/motion-generation/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 50
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 50
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')
        # split_file = pjoin(self.data_root, 'test.txt')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        new_name_list_speed = []
        data_dict_speed = {}
        #for quicker test
        # from itertools import islice
        # for name in tqdm(islice(id_list, 1000)):
        for name in tqdm(id_list):
            try:
                #TODO npz
                m_token_wo_speed_list = np.load(pjoin(tokenizer_name_tune, '%s_wo_speed.npy'%name))
                speed_list = np.load(pjoin(tokenizer_name_tune, '%s_speed.npy'%name))
                m_token_list = np.load(pjoin(tokenizer_name, '%s.npy'%name))
                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                if len(m_token_list_new) == 0:
                                    continue

                                m_token_wo_speed_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_wo_speed_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                if len(m_token_wo_speed_list_new) == 0:
                                    continue 

                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {
                                    'm_token_list': m_token_list_new,
                                    'm_token_wo_speed_list': m_token_wo_speed_list_new,
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                
                                speed_list_new = [speed[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for speed in speed_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                new_name_speed = '%s_speed_%f_%f'%(name, f_tag, to_tag)
                                data_dict_speed[new_name_speed] = {'m_token_list': speed_list_new}
                                new_name_list_speed.append(new_name_speed)

                        except:
                            pass

                if flag:
                    data_dict[name] = {
                        'm_token_list': m_token_list,
                        'm_token_wo_speed_list': m_token_wo_speed_list,
                        'text': [text_dict]
                    }
                    new_name_list.append(name)
                    data_dict_speed[name] = {'m_token_list': speed_list}
                    new_name_list_speed.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.data_dict_speed = data_dict_speed
        self.name_list_speed = new_name_list_speed

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):

        data = self.data_dict[self.name_list[item]]
        m_token_list, m_token_wo_speed_list, text_list = data['m_token_list'], data['m_token_wo_speed_list'], data['text']
        data_speed = self.data_dict_speed[self.name_list_speed[item]]
        m_token_list_speed = data_speed['m_token_list']
        assert len(m_token_list) == len(m_token_list_speed), "Token lists must have the same length"
        index = random.randint(0, len(m_token_list) - 1)
        m_tokens = m_token_list[index]
        m_tokens_wo_speed = m_token_wo_speed_list[index]
        m_tokens_speed = m_token_list_speed[index]
        text_data = text_list[index]
        caption = text_data['caption']

        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
                m_tokens_wo_speed = m_tokens_wo_speed[:-1]
            else:
                m_tokens = m_tokens[1:]
                m_tokens_wo_speed = m_tokens_wo_speed[1:]

        m_tokens_len = m_tokens.shape[0]
        if self.multi_sep:
            new_len = random.randint(20, self.max_motion_length-1)
            len_mult = math.ceil(new_len/m_tokens_len)
            m_tokens = np.tile(m_tokens, (len_mult, 1))[:new_len]
            m_tokens_wo_speed = np.tile(m_tokens_wo_speed, (len_mult, 1))[:new_len]
            m_tokens_len = new_len
            if m_tokens_len+1 < self.max_motion_length:
                m_tokens = np.concatenate([m_tokens, np.ones((1, 5), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len, 5), dtype=int) * self.mot_pad_idx], axis=0)
                m_tokens_wo_speed = np.concatenate([m_tokens_wo_speed, np.ones((1, 5), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len, 5), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens = np.concatenate([m_tokens, np.ones((1, 5), dtype=int) * self.mot_end_idx], axis=0)
                m_tokens_wo_speed = np.concatenate([m_tokens_wo_speed, np.ones((1, 5), dtype=int) * self.mot_end_idx], axis=0)


            m_tokens_speed = np.tile(m_tokens_speed, (len_mult, 1))[:new_len]
            m_tokens_speed_len = new_len
            if m_tokens_len+1 < self.max_motion_length:
                m_tokens_speed = np.concatenate([m_tokens_speed, np.zeros((self.max_motion_length-m_tokens_speed_len,8), dtype=float)], axis=0)
            else:
                m_tokens_speed = np.concatenate([m_tokens_speed, np.zeros((1,8), dtype=float)], axis=0)
        return caption, m_tokens, m_tokens_wo_speed, m_tokens_speed, m_tokens_len




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name_tune, tokenizer_name,unit_length=4,
                num_workers = 8, up_low_sep=False,multi_sep=False): 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name_tune = tokenizer_name_tune, tokenizer_name=tokenizer_name, unit_length=unit_length, up_low_sep=up_low_sep,multi_sep=multi_sep),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              )
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x