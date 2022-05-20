# -*- coding: utf-8 -*-
# @Time    : 2021/10/24 10:00
# @Author  : Yushuo Chen
# @Email   : chenyushuo@ruc.edu.cn

"""
################################################
"""

import numpy as np
import torch

from recbole.data.interaction import Interaction
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader

from multiprocessing import Manager, Pool, cpu_count


global_sampler = None


class PretrainRecDataLoader(AbstractDataLoader):
    """

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.sid_field = dataset.seq_id_field
        self.list_suffix = config['LIST_SUFFIX']
        self.iid_list_field = dataset.iid_field + self.list_suffix
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']

        self.time_field = config['TIME_FIELD']
        self.time_list_field = self.time_field + self.list_suffix
        self.location_field = config['LOCATION_FIELD']
        self.location_list_field = self.location_field + self.list_suffix
        self.feature_field = config['feature_field']
        self.feature_list_field = self.feature_field + self.list_suffix

        self.mask_ratio = config['mask_ratio']
        self.mask_token = dataset.item_num
        self.mask_field = config['MASK_FIELD']
        self.mask_prefix = config['MASK_PREFIX']
        self.mask_iid_list_field = self.mask_prefix + self.iid_list_field
        self.neg_prefix = config['NEG_PREFIX']
        self.neg_iid_list_field = self.neg_prefix + self.iid_list_field

        self.mask_segment_field = config['MASK_SEGMENT_FIELD']
        self.pos_segment_field = config['POS_SEGMENT_FIELD']
        self.neg_segment_field = config['NEG_SEGMENT_FIELD']

        global global_sampler
        global_sampler = sampler
        self.manager = Manager()
        self.num_workers = config['num_workers'] or cpu_count()
        self.pool = Pool(processes=self.num_workers)
        self.data = None
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()

        seq_ids = self.dataset.inter_feat[self.sid_field].numpy()
        item_seq = self.dataset.inter_feat[self.iid_list_field]
        item_seq_len = self.dataset.inter_feat[self.item_list_length_field]

        result_dict = self.manager.dict()
        result_dict['seq_ids'] = seq_ids
        result_dict['mask_token'] = self.mask_token
        result_dict['mask'] = mask = (torch.rand(item_seq.shape) < self.mask_ratio) & (item_seq != 0)
        mask_item_seq = torch.where(mask, self.mask_token, item_seq)
        result_dict['item_seq_len'] = item_seq_len
        result_dict['neg_item_seq'] = item_seq.clone()
        result_dict['mask_segment'] = item_seq.clone()
        result_dict['pos_segment'] = torch.full_like(item_seq, self.mask_token)
        result_dict['neg_segment'] = torch.full_like(item_seq, self.mask_token)

        split_point = np.linspace(0, len(seq_ids), self.num_workers + 1).astype(np.int64)
        args_list = [
            (result_dict, split_point[i], split_point[i + 1])
            for i in range(self.num_workers)
        ]

        self.pool.map(construct_data, args_list)

        data = {
            self.item_list_length_field: item_seq_len,
            self.time_list_field: self.dataset.inter_feat[self.time_list_field],
            self.location_list_field: self.dataset.inter_feat[self.location_list_field],
            self.feature_list_field: self.dataset.inter_feat[self.feature_list_field],
            self.mask_field: mask,
            self.mask_iid_list_field: mask_item_seq,
            self.iid_list_field: item_seq,
            self.neg_iid_list_field: result_dict['neg_item_seq'],
            self.mask_segment_field: result_dict['mask_segment'],
            self.pos_segment_field: result_dict['pos_segment'],
            self.neg_segment_field: result_dict['neg_segment'],
        }
        self.data = Interaction(data)

        return self

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.data[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data


def construct_data(raw_data):
    result_dict, begin, end = raw_data
    slc = slice(begin, end)
    seq_ids = result_dict['seq_ids']
    mask_token = result_dict['mask_token']
    mask = result_dict['mask']
    item_seq_len = result_dict['item_seq_len']
    neg_item_seq = result_dict['neg_item_seq']
    mask_segment = result_dict['mask_segment']
    pos_segment = result_dict['pos_segment']
    neg_segment = result_dict['neg_segment']

    for i, seq_id, m, seg_length in zip(range(begin, end), seq_ids[slc], mask[slc], item_seq_len[slc]):
        seg_length = int(seg_length)
        mask_num = m.sum().item()
        sample_length = torch.randint(1, 1 + seg_length // 2, (1,)).item() if seg_length >= 2 else 0

        neg_item_ids = global_sampler.sample_by_seq_ids(seq_id, mask_num + sample_length)

        neg_item_seq[i][m] = neg_item_ids[:mask_num]

        if seg_length >= 2:
            start_id = torch.randint(0, seg_length - sample_length + 1, (1,)).item()
            pos_segment[i][start_id:start_id + sample_length] = mask_segment[i][start_id:start_id + sample_length]
            neg_segment[i][start_id:start_id + sample_length] = neg_item_ids[mask_num:]
            mask_segment[i][start_id:start_id + sample_length] = mask_token
