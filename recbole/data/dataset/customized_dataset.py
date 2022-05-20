# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

from collections import defaultdict
import copy
import numpy as np
import torch

from recbole.data import get_dataloader
from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.sampler.sampler import MetaSeqSampler
from recbole.utils.enum_type import FeatureType, FeatureSource


class GRU4RecKGDataset(KGSeqDataset):

    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):

    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config['LIST_SUFFIX']
        neg_prefix = config['NEG_PREFIX']
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field])

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = self.neg_item_list[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class PretrainRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()
        self.seq_id_field = self.config['SEQ_ID_FIELD']
        self.feature_field = self.config['feature_field']
        self.candidate_feature_list = self.config['candidate_feature_list']

    def _data_processing(self):
        super()._data_processing()
        self._construct_feature_field()

    def _construct_feature_field(self):
        feature_list = []
        length = len(self.inter_feat)
        for field in self.candidate_feature_list:
            if field not in self.inter_feat:
                self.logger.warning(f'field [{field}] is not in `inter_feat`, '
                                    f'it will not used for construct `{self.feature_field}`.')
                continue
            ftype = self.field2type[field]
            if ftype != FeatureType.TOKEN and ftype != FeatureType.TOKEN_SEQ:
                self.logger.warning(f'field [{field}] is not token-like field, '
                                    f'it will not used for construct `{self.feature_field}`.')
                continue
            feature = np.zeros((length, self.num(field)), dtype=np.float32)
            field_values = self.inter_feat[field].values
            if ftype == FeatureType.TOKEN:
                feature[np.arange(length), field_values] = 1.0
            else:
                for i, v in enumerate(field_values):
                    feature[np.full_like(v, i), v] = 1.0
            feature_list.append(feature[:, 1:])
        feature_list = np.concatenate(feature_list, axis=-1)
        self.set_field_property(
            self.feature_field, FeatureType.FLOAT_SEQ, FeatureSource.INTERACTION, feature_list.shape[-1]
        )
        self.inter_feat[self.feature_field] = list(feature_list)

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        item_list_index, target_index, item_list_length = [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.seq_id_field: torch.arange(len(item_list_length)),
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                new_dict[list_field] = torch.zeros(shape, dtype=self.inter_feat[field].dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class MetaSeqDataset(SequentialDataset):
    def split_by_ratio(self, ratios, group_by=None):
        if self.task in self.meta_learning_task:
            ratios[0] += ratios[1]
            ratios[1] = 0
            return super(MetaSeqDataset, self).split_by_ratio(ratios, group_by)
        else:
            return super(MetaSeqDataset, self).split_by_ratio(ratios, group_by)

    def leave_one_out(self, group_by, leave_one_mode):
        if self.task in self.meta_learning_task:
            return super(MetaSeqDataset, self).leave_one_out(group_by, 'test_only')
        else:
            return super(MetaSeqDataset, self).leave_one_out(group_by, leave_one_mode)

    def build(self):
        task_index = defaultdict(list)
        task_fields = self.config['task_fields']
        if len(task_fields) == 1:
            task_fields = task_fields[0]
        for i, task_value in enumerate(self.inter_feat[task_fields].values):
            if isinstance(task_fields, str):
                task = self.id2token(task_fields, task_value)
            else:
                task = tuple(self.id2token(f, t) for f, t in zip(task_fields, task_value))
            task_index[task].append(i)

        meta_learning_task = set()
        for task in self.config['meta_learning_task']:
            if isinstance(task_fields, str):
                if isinstance(task, (list, tuple)):
                    assert len(task) == 1
                    task = task[0]
                meta_learning_task.add(task)
            else:
                assert len(task) == len(task_fields)
                meta_learning_task.add(tuple(task))

        train_dataloader_class = get_dataloader(self.config, 'train')
        test_dataloader_class = get_dataloader(self.config, 'test')

        train_neg_sample_args = self.config['train_neg_sample_args']
        eval_neg_sample_args = self.config['eval_neg_sample_args']

        meta_learning_dataloaders = dict()
        eval_dataloaders = dict()
        for key, value in task_index.items():
            dataset = copy.copy(self)
            dataset.inter_feat = self.inter_feat.loc[value].reset_index(drop=True)
            dataset.task = key
            dataset.meta_learning_task = meta_learning_task
            candidate_items = np.unique(dataset.inter_feat[dataset.iid_field].values)
            train_dataset, valid_dataset, test_dataset = super(MetaSeqDataset, dataset).build()
            if key in meta_learning_task:
                assert len(valid_dataset) == 0
                support_sampler = MetaSeqSampler(train_dataset, candidate_items, train_neg_sample_args['distribution'])
                support_dataloader = train_dataloader_class(self.config, train_dataset, support_sampler, shuffle=True)
                query_sampler = MetaSeqSampler(test_dataset, candidate_items, train_neg_sample_args['distribution'])
                query_dataloader = train_dataloader_class(self.config, test_dataset, query_sampler, shuffle=True)
                meta_learning_dataloaders[key] = (support_dataloader, query_dataloader)
            else:
                train_sampler = MetaSeqSampler(train_dataset, candidate_items, train_neg_sample_args['distribution'])
                train_dataloader = train_dataloader_class(self.config, train_dataset, train_sampler, shuffle=True)
                valid_sampler = MetaSeqSampler(valid_dataset, candidate_items, eval_neg_sample_args['distribution'])
                valid_dataloader = test_dataloader_class(self.config, valid_dataset, valid_sampler, shuffle=False)
                test_sampler = MetaSeqSampler(test_dataset, candidate_items, eval_neg_sample_args['distribution'])
                test_dataloader = test_dataloader_class(self.config, test_dataset, test_sampler, shuffle=False)
                eval_dataloaders[key] = (train_dataloader, valid_dataloader, test_dataloader)

        return meta_learning_dataloaders, eval_dataloaders


class MetaTrainDataset(SequentialDataset):
    def split_by_ratio(self, ratios, group_by=None):
        ratios[0] += ratios[1]
        ratios[1] = 0
        return super(MetaTrainDataset, self).split_by_ratio(ratios, group_by)

    def leave_one_out(self, group_by, leave_one_mode):
        return super(MetaTrainDataset, self).leave_one_out(group_by, 'test_only')

    def build(self):
        task_index = defaultdict(list)
        task_fields = self.config['task_fields']
        if len(task_fields) == 1:
            task_fields = task_fields[0]
        for i, task_value in enumerate(self.inter_feat[task_fields].values):
            if isinstance(task_fields, str):
                task = self.id2token(task_fields, task_value)
            else:
                task = tuple(self.id2token(f, t) for f, t in zip(task_fields, task_value))
            task_index[task].append(i)

        train_dataloader_class = get_dataloader(self.config, 'train')
        train_neg_sample_args = self.config['train_neg_sample_args']

        candidate_items = np.unique(self.inter_feat[self.iid_field].values)
        meta_learning_dataloaders = dict()
        for key, value in task_index.items():
            dataset = copy.copy(self)
            dataset.inter_feat = self.inter_feat.loc[value].reset_index(drop=True)
            train_dataset, valid_dataset, test_dataset = super(MetaTrainDataset, dataset).build()
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                continue
            assert len(valid_dataset) == 0
            support_sampler = MetaSeqSampler(train_dataset, candidate_items, train_neg_sample_args['distribution'])
            support_dataloader = train_dataloader_class(self.config, train_dataset, support_sampler, shuffle=True)
            query_sampler = MetaSeqSampler(test_dataset, candidate_items, train_neg_sample_args['distribution'])
            query_dataloader = train_dataloader_class(self.config, test_dataset, query_sampler, shuffle=True)
            meta_learning_dataloaders[key] = (support_dataloader, query_dataloader)

        return meta_learning_dataloaders


class MetaTestDataset(SequentialDataset):
    def build(self):
        task_index = defaultdict(list)
        task_fields = self.config['task_fields']
        if len(task_fields) == 1:
            task_fields = task_fields[0]
        for i, task_value in enumerate(self.inter_feat[task_fields].values):
            if isinstance(task_fields, str):
                task = self.id2token(task_fields, task_value)
            else:
                task = tuple(self.id2token(f, t) for f, t in zip(task_fields, task_value))
            task_index[task].append(i)

        train_dataloader_class = get_dataloader(self.config, 'train')
        test_dataloader_class = get_dataloader(self.config, 'test')

        train_neg_sample_args = self.config['train_neg_sample_args']
        eval_neg_sample_args = self.config['eval_neg_sample_args']

        eval_dataloaders = dict()
        candidate_items = np.arange(1, self.item_num)
        for key, value in task_index.items():
            dataset = copy.copy(self)
            dataset.inter_feat = self.inter_feat.loc[value].reset_index(drop=True)
            train_dataset, valid_dataset, test_dataset = super(MetaTestDataset, dataset).build()
            if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(test_dataset) == 0:
                continue
            train_sampler = MetaSeqSampler(train_dataset, candidate_items, train_neg_sample_args['distribution'])
            train_dataloader = train_dataloader_class(self.config, train_dataset, train_sampler, shuffle=True)
            valid_sampler = MetaSeqSampler(valid_dataset, candidate_items, eval_neg_sample_args['distribution'])
            valid_dataloader = test_dataloader_class(self.config, valid_dataset, valid_sampler, shuffle=False)
            test_sampler = MetaSeqSampler(test_dataset, candidate_items, eval_neg_sample_args['distribution'])
            test_dataloader = test_dataloader_class(self.config, test_dataset, test_sampler, shuffle=False)
            eval_dataloaders[key] = (train_dataloader, valid_dataloader, test_dataloader)

        return eval_dataloaders
