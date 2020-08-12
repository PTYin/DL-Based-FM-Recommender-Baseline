from .FM import FM
from .NFM import NFM
from .DeepFM import DeepFM
from .XDeepFM import XDeepFM
from .NGCF import NGCF
import torch
import os
import traceback


def load_model(config_model):
    """
    load saved model
    :param config_model: model parameters
    :param tag: (Optional) unique string of config doc
    :return:
    """
    print('Load Model')
    model = None
    if 'load' in config_model and config_model['load']:
        if 'load_path' in config_model and os.path.exists(config_model['load_path']):
            model = torch.load(config_model['load_path'])
        else:
            raise Exception("load model failed!")
    return model


def create_model(config_model, num_features, field_size, node_map, user_bought):
    """
    create model from config
    :param config_model: model parameters
    :param num_features: total features in all dataset
    :param field_size: number of fields of features
    :param node_map: used for constructing user-item graph of bipartite graph recommendation
    :param user_bought: used for constructing user-item graph of bipartite graph recommendation
    :return: subclass of nn.Module
    """
    print('Create Model')
    model = None
    if 'hyper_params' in config_model:
        hyper_params = config_model['hyper_params']
        if config_model['name'] == 'NFM':
            fm_model = None
            if config_model['pre_train']:
                assert os.path.exists(config_model['fm_model_path']), 'lack of FM model'
                fm_model = torch.load(config_model['fm_model_path'])

            model = NFM(num_features,
                        hyper_params['hidden_factors'],
                        hyper_params['activation_function'],
                        hyper_params['layers'],
                        hyper_params['batch_norm'],
                        hyper_params['dropout'],
                        hyper_params['lambda'],
                        fm_model)

        elif config_model['name'] == 'FM':
            model = FM(num_features,
                       hyper_params['hidden_factors'],
                       hyper_params['batch_norm'],
                       hyper_params['dropout'],
                       hyper_params['lambda'])

        elif config_model['name'] == 'DeepFM':
            model = DeepFM(num_features,
                           field_size,
                           hyper_params['embedding_size'],
                           hyper_params['deep_layers'],
                           hyper_params['dropout_fm'],
                           hyper_params['dropout_deep'],
                           hyper_params['activation_function'],
                           hyper_params['batch_norm'],
                           hyper_params['lambda'])

        elif config_model['name'] == 'XDeepFM':
            model = XDeepFM(num_features,
                            field_size,
                            hyper_params['embedding_size'],
                            hyper_params['deep_layers'],
                            hyper_params['cin_layers'],
                            hyper_params['cin_split_half'],
                            hyper_params['dropout_deep'],
                            hyper_params['deep_act'],
                            hyper_params['cin_act'],
                            hyper_params['batch_norm'],
                            hyper_params['lambda'])

        elif config_model['name'] == 'NGCF':
            model = NGCF(node_map, user_bought,
                         hyper_params['embedding_size'],
                         hyper_params['layers'],
                         hyper_params['dropout'],
                         hyper_params['normalized'],
                         hyper_params['lambda'],
                         hyper_params['activation_function'])
    else:
        raise Exception('create model failed!')
    return model
