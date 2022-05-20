# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 10:00
# @Author  : Yushuo Chen
# @Email   : chenyushuo@ruc.edu.cn

"""
################################################
"""
from collections import OrderedDict

import numpy as np
import torch

from recbole.data.interaction import Interaction
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader


class MetaLearningDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, meta_learning_dataloaders, shuffle=True):
        self.meta_learning_dataloaders = meta_learning_dataloaders
        self.task_name = list(meta_learning_dataloaders.keys())
        super(MetaLearningDataLoader, self).__init__(config, dataset, None, shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.task_name)

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__` if self.shuffle is True.
        """
        np.random.shuffle(self.task_name)

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        tasks = self.task_name[self.pr:self.pr+self.step]
        self.pr += self.step
        result = OrderedDict()
        for task in tasks:
            result[task] = self.meta_learning_dataloaders[task]
        return result
