# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
import os.path
from collections import OrderedDict
from logging import getLogger

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders, \
    MetaLearningDataLoader
from recbole.data.dataset import MetaSeqDataset, MetaTrainDataset, MetaTestDataset
from recbole.model.layers import ItemGenerator
from recbole.trainer import MetaLearningTrainer, MetaTestTrainer, MetaFusionTrainer, AttentionModuleTrainer
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def run_meta(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaSeqDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, test_data = dataset.build()
    train_data = MetaLearningDataLoader(config, dataset, train_data)
    test_data = MetaLearningDataLoader(config, dataset, test_data, shuffle=False)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = MetaLearningTrainer(config, model)

    # model training
    best_train_loss = trainer.meta_fit(train_data, saved=saved, show_progress=config['show_progress'])

    # model evaluation
    model_file_dict, test_result = trainer.meta_evaluate(
        test_data, load_best_model=saved, show_progress=config['show_progress']
    )

    logger.info(set_color('meta model file', 'yellow') + f': {trainer.saved_meta_model_file}')
    logger.info(set_color('model file dict', 'yellow') + f': {model_file_dict}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return test_result


def run_meta_train(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaTrainDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data = dataset.build()
    train_data = MetaLearningDataLoader(config, dataset, train_data)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = MetaLearningTrainer(config, model)

    # model training
    best_train_loss, task_loss = trainer.meta_fit(train_data, saved=saved, show_progress=False)

    logger.info(set_color('best train loss', 'yellow') + f': {best_train_loss}')
    logger.info(set_color('task loss', 'yellow') + f': {task_loss}')

    return trainer.saved_meta_model_file


def run_meta_test(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaTestDataset(config)
    logger.info(dataset)

    # dataset splitting
    test_data = dataset.build()
    test_data = MetaLearningDataLoader(config, dataset, test_data, shuffle=False)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, test_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer = MetaTestTrainer(config, model)

    # model evaluation
    valid_result, test_result = trainer.meta_evaluate(
        test_data, meta_model_file=config['model_file'], item_emb_file=config['item_emb_file'],
        load_best_model=saved, show_progress=config['show_progress']
    )

    def summarize_result(result):
        summarize = OrderedDict()
        for task, res in result.items():
            for key, value in res.items():
                if key not in summarize:
                    summarize[key] = 0.0
                summarize[key] += value
        for key in summarize:
            summarize[key] = summarize[key] / len(result)
        return summarize

    valid_summarize = summarize_result(valid_result)
    test_summarize = summarize_result(test_result)

    logger.info(set_color('valid summarize', 'yellow') + f': {valid_summarize}')
    logger.info(set_color('valid result', 'yellow') + f': {valid_result}')
    logger.info(set_color('test summarize', 'yellow') + f': {test_summarize}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return test_summarize


class AttentionDataset(Dataset):
    def __init__(self, last_layer, scores, pos_items, device):
        assert len(last_layer) == len(scores) == len(pos_items)
        self.length = len(last_layer)
        self.last_layer = last_layer.to(device)
        self.scores = scores.to(device)
        self.pos_items = pos_items.to(device)

    def __getitem__(self, index):
        return self.last_layer[index], self.scores[index], self.pos_items[index]

    def __len__(self):
        return self.length


def run_attention_fusion_train(
    model_name_list=None, dataset=None, config_file_lists=None, config_dict=None, saved=True
):
    # configurations initialization
    config_list = []
    for model, config_file_list in zip(model_name_list, config_file_lists):
        config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
        config_list.append(config)
    config = config_list[0]

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaTrainDataset(config)
    logger.info(dataset)

    # dataset splitting
    test_data = dataset.build()
    test_data = MetaLearningDataLoader(config, dataset, test_data, shuffle=False)

    # model loading and initialization
    model_list = []
    for model_config in config_list:
        init_seed(model_config['seed'], model_config['reproducibility'])
        model = get_model(model_config['model'])(model_config, test_data.dataset).to(model_config['device'])
        logger.info(model)
        model_list.append(model)

    # trainer loading and initialization
    trainer = AttentionModuleTrainer(config_list, model_list)

    attention_data_path = config['attention_data_path']
    if os.path.exists(attention_data_path):
        train_attention_dataloader, valid_attention_dataloader = torch.load(attention_data_path)
    else:
        # create training data
        train_last_layer, train_scores, train_item, valid_last_layer, valid_scores, valid_item = trainer.construct_data(
            test_data, meta_model_file=config['model_file'], show_progress=config['show_progress']
        )

        train_attention_dataset = AttentionDataset(train_last_layer, train_scores, train_item, config['device'])
        valid_attention_dataset = AttentionDataset(valid_last_layer, valid_scores, valid_item, config['device'])
        train_attention_dataloader = DataLoader(dataset=train_attention_dataset, batch_size=config['train_batch_size'])
        valid_attention_dataloader = DataLoader(dataset=valid_attention_dataset, batch_size=config['train_batch_size'])
        torch.save((train_attention_dataloader, valid_attention_dataloader), attention_data_path)

    best_valid_loss = trainer.attention_module_training(
        train_attention_dataloader, valid_attention_dataloader, saved=saved, show_progress=config['show_progress']
    )

    logger.info(set_color('best valid loss', 'yellow') + f': {best_valid_loss}')

    return best_valid_loss


def run_meta_fusion_test(model_name_list=None, dataset=None, config_file_lists=None, config_dict=None, saved=True):
    # configurations initialization
    config_list = []
    for model, config_file_list in zip(model_name_list, config_file_lists):
        config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
        config_list.append(config)
    config = config_list[0]

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaTestDataset(config)
    logger.info(dataset)

    # dataset splitting
    test_data = dataset.build()
    test_data = MetaLearningDataLoader(config, dataset, test_data, shuffle=False)

    # model loading and initialization
    model_list = []
    for model_config in config_list:
        init_seed(model_config['seed'], model_config['reproducibility'])
        model = get_model(model_config['model'])(model_config, test_data.dataset).to(model_config['device'])
        logger.info(model)
        model_list.append(model)

    # trainer loading and initialization
    trainer = MetaFusionTrainer(config_list, model_list)

    # model evaluation
    valid_result, test_result = trainer.meta_fusion_evaluate_with_model_file(
        test_data, meta_model_file=config['model_file'],
        load_best_model=saved, show_progress=config['show_progress']
    )

    def summarize_result(result):
        summarize = OrderedDict()
        for task, res in result.items():
            for key, value in res.items():
                if key not in summarize:
                    summarize[key] = 0.0
                summarize[key] += value
        for key in summarize:
            summarize[key] = summarize[key] / len(result)
        return summarize

    valid_summarize = summarize_result(valid_result)
    test_summarize = summarize_result(test_result)

    logger.info(set_color('valid summarize', 'yellow') + f': {valid_summarize}')
    logger.info(set_color('valid result', 'yellow') + f': {valid_result}')
    logger.info(set_color('test summarize', 'yellow') + f': {test_summarize}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return test_summarize


def run_emb_gen_train(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = MetaTrainDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data = dataset.build()
    train_data = MetaLearningDataLoader(config, dataset, train_data)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset)
    checkpoint = torch.load(config['model_file'])
    model.load_state_dict(checkpoint['state_dict'])
    model.requires_grad_(False)
    model.item_embedding = ItemGenerator(config, model.item_embedding)
    model = model.to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = MetaLearningTrainer(config, model)

    # model training
    best_train_loss, task_loss = trainer.meta_fit(train_data, saved=saved, show_progress=False)

    item_emb = model.item_embedding.generate_item_emb(torch.arange(dataset.item_num, device=config['device']))
    old_item_id = torch.from_numpy(np.unique(dataset.inter_feat[dataset.iid_field])).to(device=config['device'])
    mask = torch.ones(dataset.item_num, dtype=torch.bool, device=config['device'])
    mask[0] = mask[old_item_id] = 0
    mask = mask.unsqueeze(-1).expand_as(item_emb)
    checkpoint['state_dict']['item_embedding.weight'] = torch.where(
        mask, item_emb, checkpoint['state_dict']['item_embedding.weight']
    )
    torch.save(checkpoint, trainer.saved_meta_model_file)

    logger.info(set_color('saved meta model file', 'yellow') + f': {trainer.saved_meta_model_file}')
    logger.info(set_color('best train loss', 'yellow') + f': {best_train_loss}')
    logger.info(set_color('task loss', 'yellow') + f': {task_loss}')

    return trainer.saved_meta_model_file
