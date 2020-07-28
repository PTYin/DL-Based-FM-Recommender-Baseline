import os
import sys
import time
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

import dataset
import models
import metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='../config_example/NFM.yaml',
                        help='path for configure file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("config file doesn't exist")
        exit(-1)
    config = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)
    # print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['model']['gpu']
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(config['model']['model_path']):
        os.mkdir(config['model']['model_path'])

    # ----------------------------------Prepare Dataset----------------------------------
    print('Prepare Dataset')
    features_map = {}
    num_features = 0
    train_dataset, valid_dataset, test_dataset = None, None, None
    if config['dataset']['decoder'] == 'libfm':
        for file in config['dataset']['paths'].values():
            dataset.LibFMDataset.read_features(file, features_map)
        num_features = len(features_map)
        print("number of features:", num_features)
        train_dataset = dataset.LibFMDataset(config['dataset']['paths']['train'], features_map)
        valid_dataset = dataset.LibFMDataset(config['dataset']['paths']['valid'], features_map)
        test_dataset = dataset.LibFMDataset(config['dataset']['paths']['test'], features_map)
    train_loader = DataLoader(train_dataset, drop_last=True,
                              batch_size=config['model']['hyper_params']['batch_size'], shuffle=True,
                              num_workers=config['dataset']['num_workers'])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['model']['hyper_params']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['model']['hyper_params']['batch_size'], shuffle=False, num_workers=0)

    # ----------------------------------Create Model----------------------------------
    print('Create Model')
    model = None
    if config['model']['name'] == 'NFM':  # NFM
        fm_model = None
        if config['model']['pre_train']:
            assert os.path.exists(config['model']['fm_model_path']), 'lack of FM model'
            assert config.model == 'NFM', 'only support NFM for now'
            fm_model = torch.load(config['model']['fm_model_path'])

        model = models.NFM(num_features,
                           config['model']['hyper_params']['hidden_factors'],
                           config['model']['activation_function'],
                           config['model']['hyper_params']['layers'],
                           config['model']['batch_norm'],
                           config['model']['hyper_params']['dropout'],
                           fm_model)

    elif config['model']['name'] == 'FM':  # FM
        model = models.FM(num_features,
                          config['model']['hyper_params']['hidden_factors'],
                          config['model']['batch_norm'],
                          config['model']['hyper_params']['dropout'])

    if config['model']['load']:
        if os.path.exists(os.path.join(config['model']['model_path'], '{}.pth'.format(config['model']['name']))):
            model = torch.load(
                os.path.join(config['model']['model_path'], '{}.pth'.format(config['model']['name'])))

    model.cuda()

    # ----------------------------------Construct Optimizer----------------------------------
    print('Construct Optimizer')
    optimizer = None
    if config['model']['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config['model']['hyper_params']['learning_rate'],
                                  initial_accumulator_value=1e-8)
    elif config['model']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['model']['hyper_params']['learning_rate'])
    elif config['model']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['model']['hyper_params']['learning_rate'])
    elif config['model']['optimizer'] == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=config['model']['hyper_params']['learning_rate'], momentum=0.95)

    # ----------------------------------Construct Loss Function----------------------------------
    print('Construct Loss Function')
    if config['model']['loss_type'] == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    else:  # log_loss
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # ----------------------------------Training----------------------------------
    print('Training...')
    for epoch in range(config['model']['epochs']):
        model.train()
        start_time = time.time()
        loss = 0
        for features, feature_values, label in train_loader:
            features = features.cuda()
            feature_values = feature_values.cuda()
            label = label.cuda()

            model.zero_grad()
            prediction = model(features, feature_values)
            loss = criterion(prediction, label)
            loss += config['model']['hyper_params']['lambda'] * model.embeddings.weight.norm()
            params = list(model.parameters())
            loss.backward()
            optimizer.step()

        print("Running Epoch {:03d}/{}".format(epoch, config['model']['epochs']),
              "loss:", float(loss),
              "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
        sys.stdout.flush()

    if config['model']['save']:
        torch.save(model, os.path.join(config['model']['model_path'], '{}.pth'.format(config['model']['name'])))

    # ----------------------------------Evaluation----------------------------------
    print('Evaluating')
    if config['model']['evaluation']:
        model.eval()
        train_result = metrics.RMSE(model, train_loader)
        valid_result = metrics.RMSE(model, valid_loader)
        test_result = metrics.RMSE(model, test_loader)
        print("Train_RMSE: {:.3f}, Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(train_result, valid_result, test_result))
