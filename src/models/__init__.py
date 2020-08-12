from .FM import FM
from .NFM import NFM
from .DeepFM import DeepFM
from .XDeepFM import XDeepFM
from .NGCF import NGCF
import torch
import os


def create_model(config_model, num_features, field_size, node_map, user_bought):
    """

    :param config_model: model parameters
    :param num_features: total features in all dataset
    :param field_size: number of fields of features
    :param node_map: used for constructing user-item graph of bipartite graph recommendation
    :param user_bought: used for constructing user-item graph of bipartite graph recommendation
    :return: subclass of nn.Module
    """
    model = None
    if config_model['name'] == 'NFM':
        fm_model = None
        if config_model['pre_train']:
            assert os.path.exists(config_model['fm_model_path']), 'lack of FM model'
            fm_model = torch.load(config_model['fm_model_path'])

        model = NFM(num_features,
                    config_model['hyper_params']['hidden_factors'],
                    config_model['activation_function'],
                    config_model['hyper_params']['layers'],
                    config_model['batch_norm'],
                    config_model['hyper_params']['dropout'],
                    config_model['hyper_params']['lambda'],
                    fm_model)

    elif config_model['name'] == 'FM':
        model = FM(num_features,
                   config_model['hyper_params']['hidden_factors'],
                   config_model['batch_norm'],
                   config_model['hyper_params']['dropout'],
                   config_model['hyper_params']['lambda'])

    elif config_model['name'] == 'DeepFM':
        model = DeepFM(num_features,
                       field_size,
                       config_model['hyper_params']['embedding_size'],
                       config_model['hyper_params']['deep_layers'],
                       config_model['hyper_params']['dropout_fm'],
                       config_model['hyper_params']['dropout_deep'],
                       config_model['activation_function'],
                       config_model['batch_norm'],
                       config_model['hyper_params']['lambda'])

    elif config_model['name'] == 'XDeepFM':
        model = XDeepFM(num_features,
                        field_size,
                        config_model['hyper_params']['embedding_size'],
                        config_model['hyper_params']['deep_layers'],
                        config_model['hyper_params']['cin_layers'],
                        config_model['cin_split_half'],
                        config_model['hyper_params']['dropout_deep'],
                        config_model['deep_act'],
                        config_model['cin_act'],
                        config_model['batch_norm'],
                        config_model['hyper_params']['lambda'])

    elif config_model['name'] == 'NGCF':
        model = NGCF(node_map, user_bought,
                     config_model['hyper_params']['embedding_size'],
                     config_model['hyper_params']['layers'],
                     config_model['hyper_params']['dropout'],
                     config_model['normalized'],
                     config_model['hyper_params']['lambda'],
                     config_model['activation_function'])

    if config_model['load']:
        if os.path.exists(os.path.join(config_model['model_path'], '{}.pth'.format(config_model['name']))):
            model = torch.load(
                os.path.join(config_model['model_path'], '{}.pth'.format(config_model['name'])))

    return model
