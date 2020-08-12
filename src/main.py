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
                        default='../config_example/DeepFM.yaml',
                        help='path for configure file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("config file doesn't exist")
        exit(-1)
    config = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)
    # print(config)
    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['model']['gpu']
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(config['model']['model_path']):
        os.mkdir(config['model']['model_path'])

    # ----------------------------------Prepare Dataset----------------------------------
    print('Prepare Dataset')
    feature_map = {}
    node_map = None
    user_bought = None
    if 'bipartite' in config['dataset'] and config['dataset']['bipartite']:
        node_map = {}
        user_bought = {}
    num_features = 0
    field_size = 0
    train_dataset, valid_dataset, test_dataset = None, None, None
    if config['dataset']['decoder'] == 'libfm':
        for file in config['dataset']['paths'].values():
            field_size = dataset.LibFMDataset.read_features(file, feature_map, field_size, node_map)
        num_features = len(feature_map)
        print("number of features:", num_features)
        train_dataset = dataset.LibFMDataset(config['dataset']['paths']['train'], feature_map, node_map, user_bought)
        valid_dataset = dataset.LibFMDataset(config['dataset']['paths']['valid'], feature_map)
        test_dataset = dataset.LibFMDataset(config['dataset']['paths']['test'], feature_map)
    train_loader = DataLoader(train_dataset, drop_last=True,
                              batch_size=config['model']['hyper_params']['batch_size'], shuffle=True,
                              num_workers=config['dataset']['num_workers'])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config['model']['hyper_params']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['model']['hyper_params']['batch_size'], shuffle=False, num_workers=0)

    # ----------------------------------Create Model----------------------------------
    print('Create Model')
    model = models.create_model(config['model'], num_features, field_size, node_map, user_bought)
    assert model is not None
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
    criterion = None
    if config['model']['loss_type'] == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    elif config['model']['loss_type'] == 'log_loss':  # log_loss
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # ----------------------------------Training----------------------------------
    print('Training...')
    best_result = 100
    saved = False
    for epoch in range(config['model']['epochs']):
        model.train()
        start_time = time.time()
        loss = 0
        for features, feature_values, label in train_loader:
            features = features.cuda()
            feature_values = feature_values.cuda()
            label = label.cuda()
            if config['model']['loss_type'] == 'log_loss':
                label = label.clamp(min=0., max=1.)

            model.zero_grad()
            prediction = model(features, feature_values)
            loss = criterion(prediction, label)
            # ---------l2 regularization---------
            if model.l2 is not None:
                loss += model.l2_regularization()
            loss.backward()
            optimizer.step()

        print("Running Epoch {:03d}/{:03d} loss:{:.3f}".format(epoch, config['model']['epochs'], float(loss)),
              "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))

        # ----------------------------------Evaluation----------------------------------
        if config['model']['evaluation']:
            model.eval()
            train_result = metrics.RMSE(model, train_loader)
            valid_result = metrics.RMSE(model, valid_loader)
            test_result = metrics.RMSE(model, test_loader)
            print("\tTrain_RMSE: {:.3f}, Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(train_result, valid_result,
                                                                                       test_result))
            sys.stdout.flush()

            if test_result < best_result and config['model']['save']:
                torch.save(model, os.path.join(config['model']['model_path'], '{}.pth'.format(config['model']['name'])))
                best_result = test_result

    if not saved and config['model']['save']:
        torch.save(model, os.path.join(config['model']['model_path'], '{}.pth'.format(config['model']['name'])))

