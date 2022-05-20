# -*- coding: utf-8 -*-
# @Time    : 2021/10/24 10:00
# @Author  : Yushuo Chen
# @Email   : chenyushuo@ruc.edu.cn

r"""
################################################


"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class TimeEmbedding(nn.Module):

    def __init__(self, hidden_size, timezone=8):
        super(TimeEmbedding, self).__init__()
        self.timezone = timezone
        self.hour_embedding = nn.Embedding(24, hidden_size)
        self.day_embedding = nn.Embedding(7, hidden_size)

    def forward(self, timestamp):
        timestamp += self.timezone * 3600
        hour = self.hour_embedding((timestamp / 3600).long() % 24)
        day = self.day_embedding((timestamp / 86400).long() % 7)
        return hour + day


class TimeDelta(nn.Module):

    def __init__(self):
        super(TimeDelta, self).__init__()
        self.hour_delta_value = nn.Embedding(24, 1)
        self.hour_delta_weight = nn.Linear(1, 1)
        self.day_delta_value = nn.Linear(1, 1)

    def forward(self, timestamp):
        timestamp_delta = timestamp.unsqueeze(1) - timestamp.unsqueeze(-1)
        shape = timestamp_delta.shape
        timestamp_delta = timestamp_delta.flatten()
        hour_delta = (timestamp_delta / 3600).long() % 24
        hour_delta_value = self.hour_delta_value(hour_delta)
        day_delta = (timestamp_delta / 86400).float().floor()
        day_delta_value = self.day_delta_value(day_delta.unsqueeze(-1))
        hour_delta_weight = self.hour_delta_weight(day_delta.unsqueeze(-1))
        result = hour_delta_weight * hour_delta_value + day_delta_value
        return result.view(shape)


class PretrainRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(PretrainRec, self).__init__(config, dataset)

        # load field info
        self.LIST_SUFFIX = config['LIST_SUFFIX']
        self.FEATURE = config['feature_field']
        self.FEATURE_SEQ = self.FEATURE + self.LIST_SUFFIX
        self.TIME_SEQ = config['TIME_FIELD'] + self.LIST_SUFFIX
        self.LOC_SEQ = config['LOCATION_FIELD'] + self.LIST_SUFFIX
        self.MASK_FIELD = config['MASK_FIELD']
        self.MASK_ITEM_SEQ = config['MASK_PREFIX'] + self.ITEM_SEQ
        self.NEG_ITEM_SEQ = config['NEG_PREFIX'] + self.ITEM_SEQ

        self.MASK_SEGMENT = config['MASK_SEGMENT_FIELD']
        self.POS_SEGMENT = config['POS_SEGMENT_FIELD']
        self.NEG_SEGMENT = config['NEG_SEGMENT_FIELD']

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.timezone = config['timezone']

        self.train_stage = config['train_stage']  # pretrain or finetune
        self.pre_model_path = config['pre_model_path']  # We need this for finetune
        self.mask_ratio = config['mask_ratio']
        self.aap_weight = config['aap_weight']
        self.mip_weight = config['mip_weight']
        self.map_weight = config['map_weight']
        self.sp_weight = config['sp_weight']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load dataset info
        self.n_items = dataset.item_num + 1  # for [mask]
        self.mask_token = self.n_items - 1
        self.n_features = dataset.num(self.FEATURE)

        # define layers and loss
        # modules shared by pre-training stage and fine-tuning stage
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.time_embedding = TimeEmbedding(self.hidden_size, self.timezone)
        self.location_embedding = nn.Linear(2, self.hidden_size)
        self.feature_embedding = nn.Embedding(self.n_features, self.hidden_size)
        self.timedelta = TimeDelta()

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # modules for pretrain
        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.mip_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.map_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.sp_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.loss_fct = nn.BCELoss(reduction='none')

        # modules for finetune
        if self.loss_type == 'BPR' and self.train_stage == 'finetune':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE' and self.train_stage == 'finetune':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == 'finetune':
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        assert self.train_stage in ['pretrain', 'finetune']
        if self.train_stage == 'pretrain':
            self.apply(self._init_weights)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f'Load pretrained model from {self.pre_model_path}')
            self.load_state_dict(pretrained['state_dict'])

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _associated_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.aap_norm(sequence_output)  # [aap_num H]
        scores = sequence_output @ feature_embedding.T  # [aap_num H] @ [H feat_num] -> [aap_num feat_num]
        return torch.sigmoid(scores)

    def _mask_item_predict(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(sequence_output)  # [mask_num H]
        scores = (sequence_output * target_item_emb).sum(-1)  # [mask_num H] -> [mask_num]
        return torch.sigmoid(scores)  # [mask_num]

    def _mask_attribute_prediction(self, sequence_output, feature_embedding):
        sequence_output = self.map_norm(sequence_output)  # [mask_num H]
        scores = sequence_output @ feature_embedding.T  # [mask_num H] @ [H feat_num] -> [mask_num feat_num]
        return torch.sigmoid(scores)  # [mask_num feat_num]

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)  # [B H]
        scores = (context * segment_emb).sum(-1)  # [B H] -> [B]
        return torch.sigmoid(scores)

    def get_attention_mask(self, item_seq, time_seq, bidirectional=True):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            max_len = attention_mask.size(-1)
            attn_shape = (1, 1, max_len, max_len)
            subsequent_mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool, device=item_seq.device))
            extended_attention_mask = extended_attention_mask & subsequent_mask
        timedelta = self.timedelta(time_seq).unsqueeze(1)  # 目前效果不是很好
        extended_attention_mask = torch.where(
            extended_attention_mask, timedelta, torch.tensor(-1e5, device=time_seq.device)
        )
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        return extended_attention_mask

    def forward(self, item_seq, time_seq, loc_seq, bidirectional=True, item_seq_len=None):
        item_emb = self.item_embedding(item_seq)
        time_emb = self.time_embedding(time_seq)
        loc_emb = self.location_embedding(loc_seq)
        input_emb = item_emb + time_emb + loc_emb  # 可以再time emb和loc emb前面加个权重，这两个emb的学习效果也不是很好
        attention_mask = self.get_attention_mask(item_seq, time_seq, bidirectional=bidirectional)
        seq_output = self.trm_encoder(input_emb, attention_mask, output_all_encoded_layers=False)[0]
        if item_seq_len is not None:
            seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        return seq_output

    def pretrain(self, inter):
        masked_item_sequence = inter[self.MASK_ITEM_SEQ]
        item_seq_len = inter[self.ITEM_SEQ_LEN]
        pos_items = inter[self.ITEM_SEQ]
        neg_items = inter[self.NEG_ITEM_SEQ]
        time_sequence = inter[self.TIME_SEQ]
        location_sequence = inter[self.LOC_SEQ]
        masked_segment = inter[self.MASK_SEGMENT]
        pos_segment = inter[self.POS_SEGMENT]
        neg_segment = inter[self.NEG_SEGMENT]
        sequence_output = self.forward(masked_item_sequence, time_sequence, location_sequence)

        feature_embedding = self.feature_embedding.weight
        features = inter[self.FEATURE_SEQ]
        mask = inter[self.MASK_FIELD]

        # AAP
        aap_mask = (masked_item_sequence != 0) ^ mask  # only compute loss at non-masked position
        aap_scores = self._associated_attribute_prediction(sequence_output[aap_mask], feature_embedding)
        aap_loss = self.loss_fct(aap_scores, features[aap_mask]).sum()

        # MIP
        pos_item_embs = self.item_embedding(pos_items[mask])
        neg_item_embs = self.item_embedding(neg_items[mask])
        pos_scores = self._mask_item_predict(sequence_output[mask], pos_item_embs)
        neg_scores = self._mask_item_predict(sequence_output[mask], neg_item_embs)
        mip_distance = torch.sigmoid(pos_scores - neg_scores)
        mip_loss = self.loss_fct(mip_distance, torch.ones_like(mip_distance)).sum()

        # MAP
        map_scores = self._mask_attribute_prediction(sequence_output[mask], feature_embedding)
        map_loss = self.loss_fct(map_scores, features[mask]).sum()

        # SP
        segment_context = self.forward(masked_segment, time_sequence, location_sequence, item_seq_len=item_seq_len)
        pos_segment_emb = self.forward(pos_segment, time_sequence, location_sequence, item_seq_len=item_seq_len)
        pos_segment_scores = self._segment_prediction(segment_context, pos_segment_emb)
        neg_segment_emb = self.forward(neg_segment, time_sequence, location_sequence, item_seq_len=item_seq_len)
        neg_segment_scores = self._segment_prediction(segment_context, neg_segment_emb)
        sp_distance = torch.sigmoid(pos_segment_scores - neg_segment_scores)
        sp_loss = self.loss_fct(sp_distance, torch.ones_like(sp_distance)).sum()

        pretrain_loss = 0.0
        for loss_name in ['aap', 'mip', 'map', 'sp']:
            pretrain_loss += getattr(self, f'{loss_name}_weight') * locals()[f'{loss_name}_loss']

        return pretrain_loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            loss = self.pretrain(interaction)
        else:  # finetune
            item_seq = interaction[self.ITEM_SEQ]
            time_seq = interaction[self.TIME_SEQ]
            loc_seq = interaction[self.LOC_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            pos_items = interaction[self.POS_ITEM_ID]
            seq_output = self.forward(item_seq, time_seq, loc_seq, bidirectional=False, item_seq_len=item_seq_len)

            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_scores = (seq_output * pos_items_emb).sum(dim=-1)
                neg_scores = (seq_output * neg_items_emb).sum(dim=-1)
                loss = self.loss_fct(pos_scores, neg_scores)
            else:  # loss_type == 'CE'
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        loc_seq = interaction[self.LOC_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, time_seq, loc_seq, bidirectional=False, item_seq_len=item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = (seq_output * test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq = interaction[self.TIME_SEQ]
        loc_seq = interaction[self.LOC_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, time_seq, loc_seq, bidirectional=False, item_seq_len=item_seq_len)
        test_items_emb = self.item_embedding.weight[:-1]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.T)  # [B, n_items]
        return scores
