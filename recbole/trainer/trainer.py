# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2021/6/23, 2020/9/26, 2020/9/26, 2020/10/01, 2020/9/16
# @Author : Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan
# @Email  : zhlin@ruc.edu.cn, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/8, 2020/10/15, 2020/11/20, 2021/2/20, 2021/3/3, 2021/3/5, 2021/7/18
# @Author : Hui Wang, Xinyan Fan, Chen Yang, Yibo Li, Lanling Xu, Haoran Cheng, Zhichao Feng
# @Email  : hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn, 254170321@qq.com, 2018202152@ruc.edu.cn, xulanling_sherry@163.com, chenghaoran29@foxmail.com, fzcbupt@gmail.com

r"""
recbole.trainer.trainer
################################
"""
import copy
import os
from collections import OrderedDict
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.model.layers import MLPLayers
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, init_seed


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config['eval_type']
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learner': self.config['learner'],
            'learning_rate': self.config['learning_rate'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config['eval_type'] == EvaluatorType.RANKING:
            col_idx = interaction[self.config['ITEM_ID_FIELD']]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)


class KGTrainer(Trainer):
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super(KGTrainer, self).__init__(config, model)

        self.train_rec_step = config['train_rec_step']
        self.train_kg_step = config['train_kg_step']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step:
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
            )
        return None


class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(KGATTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # train rs
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(
            train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
        )

        # update A
        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


class PretrainTrainer(Trainer):
    r"""PretrainTrainer is designed for pre-training.
    It can be inherited by the trainer which needs pre-training and fine-tuning.
    """

    def __init__(self, config, model):
        super(PretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config['pretrain_epochs']
        self.save_step = self.config['save_step']

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def pretrain(self, train_data, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], str(epoch_idx + 1))
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = set_color('Saving current', 'blue') + ': %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result


class S3RecTrainer(PretrainTrainer):
    r"""S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
        It includes two training stages: pre-training ang fine-tuning.

        """

    def __init__(self, config, model):
        super(S3RecTrainer, self).__init__(config, model)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.model.train_stage == 'pretrain':
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == 'finetune':
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!")


class MKRTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(MKRTrainer, self).__init__(config, model)
        self.kge_interval = config['kge_interval']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        rs_total_loss, kg_total_loss = 0., 0.

        # train rs
        self.logger.info('Train RS')
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data, epoch_idx, loss_func=self.model.calculate_rs_loss, show_progress=show_progress
        )

        # train kg
        if epoch_idx % self.kge_interval == 0:
            self.logger.info('Train KG')
            train_data.set_mode(KGDataLoaderState.KG)
            kg_total_loss = super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
            )

        return rs_total_loss, kg_total_loss


class PretrainRecTrainer(PretrainTrainer):
    r"""PretrainRecTrainer is designed for PretrainRecTrainer,
    which is a self-supervised learning based sequential recommenders.
    It includes two training stages: pre-training ang fine-tuning.

    """

    def __init__(self, config, model):
        super(PretrainRecTrainer, self).__init__(config, model)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.model.train_stage == 'pretrain':
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == 'finetune':
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune'!")


class TraditionalTrainer(Trainer):
    r"""TraditionalTrainer is designed for Traditional model(Pop,ItemKNN), which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model):
        super(TraditionalTrainer, self).__init__(config, model)
        self.epochs = 1  # Set the epoch to 1 when running memory based model


class DecisionTreeTrainer(AbstractTrainer):
    """DecisionTreeTrainer is designed for DecisionTree model.

    """

    def __init__(self, config, model):
        super(DecisionTreeTrainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.label_field = config['LABEL_FIELD']
        self.convert_token_to_onehot = self.config['convert_token_to_onehot']

        # evaluator
        self.eval_type = config['eval_type']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.valid_metric = config['valid_metric'].lower()
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)

        # model saved
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        temp_file = '{}-{}-temp.pth'.format(self.config['model'], get_local_time())
        self.temp_file = os.path.join(self.checkpoint_dir, temp_file)

        temp_best_file = '{}-{}-temp-best.pth'.format(self.config['model'], get_local_time())
        self.temp_best_file = os.path.join(self.checkpoint_dir, temp_best_file)

        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.stopping_step = config['stopping_step']
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None

    def _interaction_to_sparse(self, dataloader):
        r"""Convert data format from interaction to sparse or numpy

        Args:
            dataloader (DecisionTreeDataLoader): DecisionTreeDataLoader dataloader.
        Returns:
            cur_data (sparse or numpy): data.
            interaction_np[self.label_field] (numpy): label.
        """
        interaction = dataloader.dataset[:]
        interaction_np = interaction.numpy()
        cur_data = np.array([])
        columns = []
        for key, value in interaction_np.items():
            value = np.resize(value, (value.shape[0], 1))
            if key != self.label_field:
                columns.append(key)
                if cur_data.shape[0] == 0:
                    cur_data = value
                else:
                    cur_data = np.hstack((cur_data, value))

        if self.convert_token_to_onehot:
            from scipy import sparse
            from scipy.sparse import dok_matrix
            convert_col_list = dataloader.dataset.convert_col_list
            hash_count = dataloader.dataset.hash_count

            new_col = cur_data.shape[1] - len(convert_col_list)
            for key, values in hash_count.items():
                new_col = new_col + values
            onehot_data = dok_matrix((cur_data.shape[0], new_col))

            cur_j = 0
            new_j = 0

            for key in columns:
                if key in convert_col_list:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, int(new_j + cur_data[i, cur_j])] = 1
                    new_j = new_j + hash_count[key] - 1
                else:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, new_j] = cur_data[i, cur_j]
                cur_j = cur_j + 1
                new_j = new_j + 1

            cur_data = sparse.csc_matrix(onehot_data)

        return cur_data, interaction_np[self.label_field]

    def _interaction_to_lib_datatype(self, dataloader):
        pass

    def _valid_epoch(self, valid_data):
        r"""

        Args:
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        valid_result = self.evaluate(valid_data, load_best_model=False)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.temp_best_file,
            'other_parameter': None
        }
        torch.save(state, self.saved_model_file)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False):
        for epoch_idx in range(self.epochs):
            self._train_at_once(train_data, valid_data)

            if (epoch_idx + 1) % self.eval_step == 0:
                # evaluate
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self.model.save_model(self.temp_best_file)
                        self._save_checkpoint(epoch_idx)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if self.temp_file:
                        os.remove(self.temp_file)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        raise NotImplementedError

    def _train_at_once(self, train_data, valid_data):
        raise NotImplementedError


class xgboostTrainer(DecisionTreeTrainer):
    """xgboostTrainer is designed for XGBOOST.

    """

    def __init__(self, config, model):
        super(xgboostTrainer, self).__init__(config, model)

        self.xgb = __import__('xgboost')
        self.boost_model = config['xgb_model']
        self.silent = config['xgb_silent']
        self.nthread = config['xgb_nthread']

        # train params
        self.params = config['xgb_params']
        self.num_boost_round = config['xgb_num_boost_round']
        self.evals = ()
        self.early_stopping_rounds = config['xgb_early_stopping_rounds']
        self.evals_result = {}
        self.verbose_eval = config['xgb_verbose_eval']
        self.callbacks = None
        self.deval = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to DMatrix

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.
        Returns:
            DMatrix: Data in the form of 'DMatrix'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.xgb.DMatrix(data=data, label=label, silent=self.silent, nthread=self.nthread)

    def _train_at_once(self, train_data, valid_data):
        r"""

        Args:
            train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        self.model = self.xgb.train(
            self.params,
            self.dtrain,
            self.num_boost_round,
            self.evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=self.verbose_eval,
            xgb_model=self.boost_model,
            callbacks=self.callbacks
        )

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model.load_model(checkpoint_file)

        self.deval = self._interaction_to_lib_datatype(eval_data)
        self.eval_true = torch.Tensor(self.deval.get_label())
        self.eval_pred = torch.Tensor(self.model.predict(self.deval))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class lightgbmTrainer(DecisionTreeTrainer):
    """lightgbmTrainer is designed for lightgbm.

    """

    def __init__(self, config, model):
        super(lightgbmTrainer, self).__init__(config, model)

        self.lgb = __import__('lightgbm')
        self.boost_model = config['lgb_model']
        self.silent = config['lgb_silent']

        # train params
        self.params = config['lgb_params']
        self.num_boost_round = config['lgb_num_boost_round']
        self.evals = ()
        self.early_stopping_rounds = config['lgb_early_stopping_rounds']
        self.evals_result = {}
        self.verbose_eval = config['lgb_verbose_eval']
        self.learning_rates = config['lgb_learning_rates']
        self.callbacks = None
        self.deval_data = self.deval_label = None
        self.eval_pred = self.eval_true = None

    def _interaction_to_lib_datatype(self, dataloader):
        r"""Convert data format from interaction to Dataset

        Args:
            dataloader (DecisionTreeDataLoader): xgboost dataloader.
        Returns:
            dataset(lgb.Dataset): Data in the form of 'lgb.Dataset'.
        """
        data, label = self._interaction_to_sparse(dataloader)
        return self.lgb.Dataset(data=data, label=label, silent=self.silent)

    def _train_at_once(self, train_data, valid_data):
        r"""

        Args:
            train_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
            valid_data (DecisionTreeDataLoader): DecisionTreeDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_lib_datatype(train_data)
        self.dvalid = self._interaction_to_lib_datatype(valid_data)
        self.evals = [self.dtrain, self.dvalid]
        self.model = self.lgb.train(
            self.params,
            self.dtrain,
            self.num_boost_round,
            self.evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=self.verbose_eval,
            learning_rates=self.learning_rates,
            init_model=self.boost_model,
            callbacks=self.callbacks
        )

        self.model.save_model(self.temp_file)
        self.boost_model = self.temp_file

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.temp_best_file
            self.model = self.lgb.Booster(model_file=checkpoint_file)

        self.deval_data, self.deval_label = self._interaction_to_sparse(eval_data)
        self.eval_true = torch.Tensor(self.deval_label)
        self.eval_pred = torch.Tensor(self.model.predict(self.deval_data))

        self.eval_collector.eval_collect(self.eval_pred, self.eval_true)
        result = self.evaluator.evaluate(self.eval_collector.get_data_struct())
        return result


class RaCTTrainer(PretrainTrainer):
    r"""RaCTTrainer is designed for RaCT, which is an actor-critic reinforcement learning based general recommenders.
        It includes three training stages: actor pre-training, critic pre-training and actor-critic training.

        """

    def __init__(self, config, model):
        super(RaCTTrainer, self).__init__(config, model)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.model.train_stage == 'actor_pretrain':
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == "critic_pretrain":
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == 'finetune':
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError(
                "Please make sure that the 'train_stage' is "
                "'actor_pretrain', 'critic_pretrain' or 'finetune'!"
            )


class RecVAETrainer(Trainer):
    r"""RecVAETrainer is designed for RecVAE, which is a general recommender.

    """

    def __init__(self, config, model):
        super(RecVAETrainer, self).__init__(config, model)
        self.n_enc_epochs = config['n_enc_epochs']
        self.n_dec_epochs = config['n_dec_epochs']

        self.optimizer_encoder = self._build_optimizer(params=self.model.encoder.parameters())
        self.optimizer_decoder = self._build_optimizer(params=self.model.decoder.parameters())

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.optimizer = self.optimizer_encoder
        encoder_loss_func = lambda data: self.model.calculate_loss(data, encoder_flag=True)
        for epoch in range(self.n_enc_epochs):
            super()._train_epoch(train_data, epoch_idx, loss_func=encoder_loss_func, show_progress=show_progress)

        self.model.update_prior()
        loss = 0.0
        self.optimizer = self.optimizer_decoder
        decoder_loss_func = lambda data: self.model.calculate_loss(data, encoder_flag=False)
        for epoch in range(self.n_dec_epochs):
            loss += super()._train_epoch(
                train_data, epoch_idx, loss_func=decoder_loss_func, show_progress=show_progress
            )
        return loss


class MetaLearningTrainer(Trainer):
    def __init__(self, config, model):
        super(MetaLearningTrainer, self).__init__(config, model)
        self.best_train_loss = np.inf
        saved_model_file = f'{self.config["model"]}-meta-{get_local_time()}.pth'
        self.saved_meta_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.meta_epochs = config['meta_epochs']
        self.num_local_update = config['num_local_update']
        self.local_learner = config['local_learner']
        self.local_learning_rate = config['local_learning_rate']
        self.local_weight_decay = config['local_weight_decay']
        model.logger = None
        self.local_model = copy.deepcopy(model)
        model.logger = self.local_model.logger = self.logger
        self.local_modules = set(config['local_modules'])
        for name, module in self.local_model.named_children():
            if name not in self.local_modules:
                module.requires_grad_(False)  # Freeze global parameter.
        self.local_optimizer = self._build_optimizer(
            params=self.local_model.parameters(),
            learner=self.local_learner,
            learning_rate=self.local_learning_rate,
            weight_decay=self.local_weight_decay
        )

    def _support_epoch(self, support_data, task, loss_func=None, show_progress=False):
        loss_func = loss_func or self.local_model.calculate_loss
        iter_data = (
            tqdm(
                support_data,
                total=len(support_data),
                ncols=100,
                desc=set_color(f"Support task {task:>3}", 'pink'),
            ) if show_progress else support_data
        )
        total_loss = 0.0
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.local_optimizer.zero_grad()
            loss = loss_func(interaction)
            if isinstance(loss, tuple):
                loss = sum(loss)
            self._check_nan(loss)
            total_loss += loss.item()
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.local_model.parameters(), **self.clip_grad_norm)
            self.local_optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def _query_epoch(self, query_data, task, loss_func=None, show_progress=False):
        loss_func = loss_func or self.model.calculate_loss
        iter_data = (
            tqdm(
                query_data,
                total=len(query_data),
                ncols=100,
                desc=set_color(f"Query task {task:>5}", 'pink'),
            ) if show_progress else query_data
        )
        total_inter_num = query_data.pr_end
        total_loss = 0.0
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
            else:
                loss = losses
            self._check_nan(loss)
            total_loss = total_loss + loss * len(interaction) / total_inter_num
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def _meta_train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        self.local_model.train()
        total_loss = 0.0
        task_loss = OrderedDict()
        for batch_idx, batch_task in enumerate(train_data):
            global_state_dict = copy.deepcopy(self.model.state_dict())
            self.optimizer.zero_grad()
            for task, (support_data, query_data) in batch_task.items():
                self.local_model.load_state_dict(global_state_dict)
                for _ in range(self.num_local_update):
                    self._support_epoch(support_data, task, loss_func, show_progress)
                for name in self.local_modules:
                    state_dict = getattr(self.local_model, name).state_dict()
                    getattr(self.model, name).load_state_dict(state_dict)
                global_loss = self._query_epoch(query_data, task, loss_func, show_progress)
                task_loss[task] = global_loss.item()
                global_loss = global_loss / len(batch_task)
                self._check_nan(global_loss)
                total_loss += global_loss.item()
                global_loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
        return total_loss, task_loss

    def meta_fit(self, train_data, verbose=True, saved=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.meta_epochs):
            # train
            training_start_time = time()
            train_loss, task_loss = self._meta_train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            self.best_train_loss, self.cur_step, stop_flag, update_flag = early_stopping(
                train_loss,
                self.best_train_loss,
                self.cur_step,
                max_step=self.stopping_step,
                bigger=False,
            )
            if saved:
                saved_model_file = f'{self.config["model"]}-meta-{epoch_idx}-{get_local_time()}.pth'
                saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
                self._save_checkpoint(epoch_idx, verbose=verbose, saved_model_file=saved_model_file)

        return self.best_train_loss, task_loss

    def _full_sort_batch_eval(self, batched_data):
        interaction, scores, positive_u, positive_i = \
            super(MetaLearningTrainer, self)._full_sort_batch_eval(batched_data)
        scores[:, self.irrelevant_mask] = -np.inf
        return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint = self.checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))

        self.model.eval()

        self.tot_item_num = eval_data.dataset.item_num
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.irrelevant_mask = torch.ones(self.tot_item_num, dtype=torch.bool, device=self.config['device'])
            self.irrelevant_mask[eval_data.sampler.candidate_items] = False
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result

    def meta_evaluate(
        self, eval_data, verbose=True, saved=True, load_best_model=True, show_progress=False
    ):
        checkpoint = torch.load(self.saved_meta_model_file)
        model_file_dict = OrderedDict()
        task_result = OrderedDict()
        for batch_idx, batch_task in enumerate(eval_data):
            for task, (train_data, valid_data, test_data) in batch_task.items():
                self.logger.info(set_color(f'Training task {task}.', 'pink'))
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.load_other_parameter(checkpoint.get('other_parameter'))
                saved_model_file = f'{self.config["model"]}-task-{task}-{get_local_time()}.pth'
                self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
                self.start_epoch = 0
                self.cur_step = 0
                self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
                self.best_valid_result = None
                self.train_loss_dict = dict()
                self.fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress)
                result = self.evaluate(test_data, load_best_model=load_best_model, show_progress=show_progress)
                self.logger.info(set_color('test result', 'yellow') + f': {result}')
                model_file_dict[task] = self.saved_model_file
                task_result[task] = result

        return model_file_dict, task_result


class MetaTestTrainer(Trainer):
    def __init__(self, config, model):
        super(MetaTestTrainer, self).__init__(config, model)

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.checkpoint = state

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint = self.checkpoint
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))

        self.model.eval()

        self.tot_item_num = eval_data.dataset.item_num
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        for batch_idx, batched_data in enumerate(eval_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result

    def meta_evaluate(
        self, eval_data, meta_model_file=None, item_emb_file=None, saved=True, load_best_model=True, show_progress=False
    ):
        checkpoint = torch.load(meta_model_file)
        # if item_emb_file is not None:
        #     item_emb_checkpoint = torch.load(item_emb_file)
        #     item_emb_weight = item_emb_checkpoint['state_dict']['item_embedding.weight']
        # else:
        #     state_dict = self.model.state_dict()
        #     item_emb_weight = state_dict['item_embedding.weight']
        # checkpoint['state_dict']['item_embedding.weight'] = item_emb_weight
        task_valid_result = OrderedDict()
        task_test_result = OrderedDict()
        iter_data = (
            tqdm(
                eval_data.meta_learning_dataloaders.items(),
                total=len(eval_data.meta_learning_dataloaders),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data.meta_learning_dataloaders.items()
        )
        for task, (train_data, valid_data, test_data) in iter_data:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            self.start_epoch = 0
            self.cur_step = 0
            self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
            self.best_valid_result = None
            self.train_loss_dict = dict()
            init_seed(self.config['seed'], self.config['reproducibility'])
            _, valid_result = self.fit(train_data, valid_data, verbose=False, saved=saved, show_progress=False)
            test_result = self.evaluate(test_data, load_best_model=load_best_model, show_progress=False)
            task_valid_result[task] = valid_result
            task_test_result[task] = test_result
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return task_valid_result, task_test_result


class WeightFusionModel(nn.Module):
    def __init__(self, config, model_list):
        super(WeightFusionModel, self).__init__()
        self.config = config
        self.fusion_weight = torch.tensor(config['fusion_weight'], device=config['device'])
        self.model_list = nn.ModuleList(model_list)

    def load_checkpoint(self, checkpoint_list):
        for model, checkpoint in zip(self.model_list, checkpoint_list):
            model.load_state_dict(checkpoint['state_dict'])
            model.load_other_parameter(checkpoint.get('other_parameter'))

    def fusion(self, scores_list):
        new_weight_shape = [1] * scores_list.dim()
        new_weight_shape[1] = -1
        weight = self.fusion_weight.view(*new_weight_shape)
        scores_list = (scores_list - scores_list.mean(-1, keepdim=True)) / scores_list.std(-1, keepdim=True)
        return (weight * scores_list).sum(1)

    def predict(self, interaction):
        scores_list = torch.stack([model.predict(interaction) for model in self.model_list], dim=1)
        return self.fusion(scores_list)

    def full_sort_predict(self, interaction):
        scores_list = torch.stack([model.full_sort_predict(interaction) for model in self.model_list], dim=1)
        return self.fusion(scores_list)


class AttentionModule(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, scores_size):
        super(AttentionModule, self).__init__()
        self.input_layer_norm = nn.LayerNorm(input_size)
        self.mlp = MLPLayers([input_size] + mlp_hidden_size + [1])
        self.softmax = nn.Softmax(dim=-2)
        self.scores_layer_norm = nn.LayerNorm(scores_size)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, last_layer_list, scores_list):  # tensor_list: [B 3 E]  scores_list: [B 3 I]
        last_layer_list = self.input_layer_norm(last_layer_list)  # [B 3 E]
        attention = self.mlp(last_layer_list)  # [B 3 1]
        attention = self.softmax(attention)  # [B 3 1]
        scores_list = self.scores_layer_norm(scores_list)  # [B 3 I]
        fusion_score = (attention * scores_list).sum(1)  # [B I]
        return fusion_score

    def calculate_loss(self, last_layer_list, scores_list, pos_items):
        logits = self.forward(last_layer_list, scores_list)
        loss = self.loss_fct(logits, pos_items)
        return loss


class AttentionFusionModel(nn.Module):
    def __init__(self, config, model_list):
        super(AttentionFusionModel, self).__init__()
        self.config = config
        self.attention_module = AttentionModule(
            config['MAX_ITEM_LIST_LENGTH'] * config['hidden_size'], config['mlp_hidden_size'], model_list[0].n_items
        )
        attention_module_checkpoint = torch.load(config['attention_module_file'])
        self.attention_module.load_state_dict(attention_module_checkpoint['state_dict'])
        self.attention_module = self.attention_module.to(config['device'])
        self.model_list = nn.ModuleList(model_list)

    def load_checkpoint(self, checkpoint_list):
        for model, checkpoint in zip(self.model_list, checkpoint_list):
            model.load_state_dict(checkpoint['state_dict'])
            model.load_other_parameter(checkpoint.get('other_parameter'))

    def fusion(self, last_layer_list, scores_list):
        return self.attention_module(last_layer_list, scores_list)

    def predict(self, interaction):
        last_layer_list, scores_list = [], []
        for model in self.model_list:
            scores, last_layer = model.predict(interaction, with_last_layer=True)
            last_layer_list.append(last_layer.view(len(last_layer), -1))  # [B E]
            scores_list.append(scores)  # [B I]
        last_layer_list = torch.stack(last_layer_list, dim=1)
        scores_list = torch.stack(scores_list, dim=1)
        return self.fusion(last_layer_list, scores_list)

    def full_sort_predict(self, interaction):
        last_layer_list, scores_list = [], []
        for model in self.model_list:
            scores, last_layer = model.full_sort_predict(interaction, with_last_layer=True)
            last_layer_list.append(last_layer.view(len(last_layer), -1))  # [B E]
            scores_list.append(scores)  # [B I]
        last_layer_list = torch.stack(last_layer_list, dim=1)
        scores_list = torch.stack(scores_list, dim=1)
        return self.fusion(last_layer_list, scores_list)


class AttentionModuleTrainer(Trainer):
    def __init__(self, config_list, model_list):
        config = config_list[0]
        self.fusion_model = WeightFusionModel(config, model_list)
        super(AttentionModuleTrainer, self).__init__(config, self.fusion_model)
        params = [
            {
                'params': model.parameters(),
                'lr': model_config['learning_rate'],
            }
            for model, model_config in zip(self.fusion_model.model_list, config_list)
        ]
        self.optimizer = self._build_optimizer(params=params)
        self.construct_data_rate = config['construct_data_rate'] or 1.0
        self.attention_module_file = config['attention_module_file']
        self.attention_module = AttentionModule(
            config['MAX_ITEM_LIST_LENGTH'] * config['hidden_size'], config['mlp_hidden_size'], model_list[0].n_items
        )
        self.attention_module = self.attention_module.to(config['device'])
        self.attention_optimizer = self._build_optimizer(
            params=self.attention_module.parameters(),
            learner=config['attention_learner'],
            learning_rate=config['attention_learning_rate'],
            weight_decay=config['attention_weight_decay'],
        )
        self.attention_start_epoch = 0
        self.attention_epochs = config['attention_epochs']
        self.best_valid_loss = np.inf

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.checkpoint = state

    @torch.no_grad()
    def _get_last_layer(self, train_data):
        self.model.eval()
        checkpoint = self.checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))
        last_layer_list, scores_list, target_item_list = [], [], []
        for batch_idx, interaction in enumerate(train_data):
            interaction = interaction.to(self.device)
            target_item = interaction[self.model.ITEM_ID]
            scores, last_layer = self.model.full_sort_predict(interaction, with_last_layer=True)
            last_layer_list.append(last_layer)
            scores_list.append(scores)
            target_item_list.append(target_item)
        last_layer = torch.cat(last_layer_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        target_item = torch.cat(target_item_list, dim=0)
        return last_layer.cpu(), scores.cpu(), target_item.cpu()  # [B E], [B I], [B]

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        tmp_train_data = copy.deepcopy(train_data)
        tmp_valid_data = copy.deepcopy(valid_data)
        train_last_layer_list, valid_last_layer_list = [], []
        train_scores_list, valid_scores_list = [], []
        train_item = valid_item = None
        for model in self.fusion_model.model_list:
            self.model = model
            self.start_epoch = 0
            self.cur_step = 0
            self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
            self.best_valid_result = None
            self.train_loss_dict = dict()
            train_data = copy.deepcopy(tmp_train_data)
            valid_data = copy.deepcopy(tmp_valid_data)
            init_seed(self.config['seed'], self.config['reproducibility'])
            _, _ = super(AttentionModuleTrainer, self).fit(
                train_data, None, verbose, saved, show_progress, callback_fn
            )
            train_last_layer, train_scores, train_item = self._get_last_layer(train_data)
            valid_last_layer, valid_scores, valid_item = self._get_last_layer(valid_data)
            train_last_layer_list.append(train_last_layer)
            valid_last_layer_list.append(valid_last_layer)
            train_scores_list.append(train_scores)
            valid_scores_list.append(valid_scores)
        train_last_layer = torch.stack(train_last_layer_list, dim=1)  # [B 3 E]
        valid_last_layer = torch.stack(valid_last_layer_list, dim=1)  # [B 3 E]
        train_scores = torch.stack(train_scores_list, dim=1)  # [B 3 I]
        valid_scores = torch.stack(valid_scores_list, dim=1)  # [B 3 I]
        return train_last_layer, train_scores, train_item, valid_last_layer, valid_scores, valid_item

    def construct_data(
        self, train_data, meta_model_file=None, saved=True, show_progress=False
    ):
        init_checkpoint_list = [torch.load(file) for file in meta_model_file]
        iter_data = (
            tqdm(
                train_data.meta_learning_dataloaders.items(),
                total=len(train_data.meta_learning_dataloaders),
                ncols=100,
                desc=set_color(f"Construct   ", 'pink'),
            ) if show_progress else train_data.meta_learning_dataloaders.items()
        )
        train_last_layer_list, valid_last_layer_list = [], []
        train_scores_list, valid_scores_list = [], []
        train_item_list, valid_item_list = [], []
        selected = (torch.rand(len(iter_data)) <= self.construct_data_rate)
        for i, (task, (support_data, query_data)) in enumerate(iter_data):
            if not selected[i]:
                continue
            self.fusion_model.load_checkpoint(init_checkpoint_list)
            train_last_layer, train_scores, train_item, valid_last_layer, valid_scores, valid_item = self.fit(
                support_data, query_data, verbose=False, saved=saved, show_progress=False
            )
            train_last_layer_list.append(train_last_layer)
            valid_last_layer_list.append(valid_last_layer)
            train_scores_list.append(train_scores)
            valid_scores_list.append(valid_scores)
            train_item_list.append(train_item)
            valid_item_list.append(valid_item)
        train_last_layer = torch.cat(train_last_layer_list, dim=0)
        valid_last_layer = torch.cat(valid_last_layer_list, dim=0)
        train_scores = torch.cat(train_scores_list, dim=0)
        valid_scores = torch.cat(valid_scores_list, dim=0)
        train_item = torch.cat(train_item_list, dim=0)
        valid_item = torch.cat(valid_item_list, dim=0)
        return train_last_layer, train_scores, train_item, valid_last_layer, valid_scores, valid_item

    def _save_attention_module(self, epoch, verbose=True, **kwargs):
        saved_model_file = kwargs.pop('saved_model_file', self.attention_module_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.attention_module.state_dict(),
            'optimizer': self.attention_optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')

    def _attention_train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.attention_module.train()
        total_loss = 0.0
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, (last_layer, scores, pos_item) in enumerate(iter_data):
            last_layer = last_layer.to(self.device)
            scores = scores.to(self.device)
            pos_item = pos_item.to(self.device)
            self.attention_optimizer.zero_grad()
            loss = self.attention_module.calculate_loss(last_layer, scores, pos_item)
            total_loss = total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.attention_module.parameters(), **self.clip_grad_norm)
            self.attention_optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    @torch.no_grad()
    def _attention_valid_epoch(self, valid_data, show_progress=False):
        self.attention_module.eval()
        total_loss = 0.0
        iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else valid_data
        )
        for batch_idx, (last_layer, scores, pos_item) in enumerate(iter_data):
            last_layer = last_layer.to(self.device)
            scores = scores.to(self.device)
            pos_item = pos_item.to(self.device)
            loss = self.attention_module.calculate_loss(last_layer, scores, pos_item)
            total_loss = total_loss + loss.item()
            self._check_nan(loss)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def attention_module_training(
        self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False
    ):
        if saved and self.attention_start_epoch >= self.attention_epochs:
            self._save_attention_module(-1, verbose=verbose)

        for epoch_idx in range(self.attention_start_epoch, self.attention_epochs):
            # train
            training_start_time = time()
            train_loss = self._attention_train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_attention_module(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_loss = self._attention_valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_loss, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_loss,
                    self.best_valid_loss,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=False,
                )
                valid_end_time = time()
                valid_loss_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_loss)
                if verbose:
                    self.logger.info(valid_loss_output)
                self.tensorboard.add_scalar('Vaild_loss', valid_loss, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_attention_module(epoch_idx, verbose=verbose)

                if stop_flag:
                    stop_output = 'Finished training, best eval loss in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_loss)
        return self.best_valid_loss


class MetaFusionTrainer(Trainer):
    def __init__(self, config_list, model_list):
        config = config_list[0]
        if config['fusion_method'] != 'attention_method':
            self.fusion_model = WeightFusionModel(config, model_list)
        else:
            self.fusion_model = AttentionFusionModel(config, model_list)
        super(MetaFusionTrainer, self).__init__(config, self.fusion_model)
        params = [
            {
                'params': model.parameters(),
                'lr': model_config['learning_rate'],
            }
            for model, model_config in zip(self.fusion_model.model_list, config_list)
        ]
        self.optimizer = self._build_optimizer(params=params)
        self.checkpoint_list = []

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.checkpoint = state

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        self.checkpoint_list = []
        tmp_train_data = copy.deepcopy(train_data)
        tmp_valid_data = copy.deepcopy(valid_data)
        for model in self.fusion_model.model_list:
            self.model = model
            self.start_epoch = 0
            self.cur_step = 0
            self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
            self.best_valid_result = None
            self.train_loss_dict = dict()
            train_data = copy.deepcopy(tmp_train_data)
            valid_data = copy.deepcopy(tmp_valid_data)
            init_seed(self.config['seed'], self.config['reproducibility'])
            _, _ = super(MetaFusionTrainer, self).fit(
                train_data, valid_data, verbose, saved, show_progress, callback_fn
            )
            self.checkpoint_list.append(self.checkpoint)
        self.model = self.fusion_model
        return self._valid_epoch(tmp_valid_data, show_progress)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            assert isinstance(self.model, (WeightFusionModel, AttentionFusionModel))
            self.model.load_checkpoint(self.checkpoint_list)

        self.model.eval()

        self.tot_item_num = eval_data.dataset.item_num
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result

    def meta_fusion_evaluate_with_model_file(
        self, eval_data, meta_model_file=None, saved=True, load_best_model=True, show_progress=False
    ):
        init_checkpoint_list = [torch.load(file) for file in meta_model_file]
        task_valid_result = OrderedDict()
        task_test_result = OrderedDict()
        iter_data = (
            tqdm(
                eval_data.meta_learning_dataloaders.items(),
                total=len(eval_data.meta_learning_dataloaders),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data.meta_learning_dataloaders.items()
        )
        for task, (train_data, valid_data, test_data) in iter_data:
            self.fusion_model.load_checkpoint(init_checkpoint_list)
            _, valid_result = self.fit(train_data, valid_data, verbose=False, saved=saved, show_progress=False)
            test_result = self.evaluate(test_data, load_best_model=load_best_model, show_progress=False)
            task_valid_result[task] = valid_result
            task_test_result[task] = test_result

        return task_valid_result, task_test_result
