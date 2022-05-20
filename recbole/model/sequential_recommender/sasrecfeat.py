# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:32
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
SASRecF
################################################
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss


class SASRecFeat(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecFeat, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        # self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        item_emb = torch.load(config['item_feat_emb'])
        if isinstance(item_emb, dict):
            item_feat = torch.zeros((dataset.item_num, item_emb['embs'].size(-1)))
            token2id = dataset.field2token_id[dataset.iid_field]
            for item, emb in zip(item_emb['item_id'], item_emb['embs']):
                if item in token2id:
                    item_id = token2id[item]
                    item_feat[item_id] = emb
        else:
            item_feat = item_emb
        self.item_feat = nn.Parameter(item_feat.to(config['device']))
        self.item_projection = nn.Linear(self.item_feat.size(-1), self.hidden_size)
        self.restore_item_e = None

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['restore_item_e']

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

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len, with_last_layer=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_projection(self.item_feat[item_seq])
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        last_layer = trm_output[-1]
        output = self.gather_indexes(last_layer, item_seq_len - 1)
        if with_last_layer:
            return output, last_layer
        return output  # [B H]

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_item_e is not None:
            self.restore_item_e = None
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_projection(self.item_feat[pos_items])
            neg_items_emb = self.item_projection(self.item_feat[neg_items])
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_projection(self.item_feat)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction, with_last_layer=False):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, with_last_layer=with_last_layer)
        if with_last_layer:
            seq_output, last_layer = seq_output
        test_item_emb = self.item_projection(self.item_feat[test_item])
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        if with_last_layer:
            return scores, last_layer.view(len(last_layer), -1)
        return scores

    def full_sort_predict(self, interaction, with_last_layer=False):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, with_last_layer=with_last_layer)
        if with_last_layer:
            seq_output, last_layer = seq_output
        if self.restore_item_e is None:
            self.restore_item_e = self.item_projection(self.item_feat)
        test_items_emb = self.restore_item_e
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        if with_last_layer:
            return scores, last_layer.view(len(last_layer), -1)
        return scores
