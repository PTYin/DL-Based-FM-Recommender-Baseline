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


def run(config):
    os.environ['DGLBACKEND'] = 'pytorch'
    if config['model']['gpu'] != 'any':
        os.environ['CUDA_VISIBLE_DEVICES'] = config['model']['gpu']
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(config['model']['model_path']):
        os.makedirs(config['model']['model_path'])

    # ----------------------------------Prepare Dataset----------------------------------
    print('Prepare Dataset')
    sys.stdout.flush()
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
        if 'valid' in config['dataset']['paths']:
            valid_dataset = dataset.LibFMDataset(config['dataset']['paths']['valid'], feature_map)
        test_dataset = dataset.LibFMDataset(config['dataset']['paths']['test'], feature_map)
    train_loader = DataLoader(train_dataset, drop_last=True,
                              batch_size=config['dataset']['batch_size'], shuffle=True,
                              num_workers=config['dataset']['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=0)

    # ----------------------------------Load or Create Model----------------------------------
    model = models.load_model(config['model'])  # load
    if model is None:  # create
        model = models.create_model(config['model'], num_features, field_size, node_map, user_bought)
    assert model is not None
    model.cuda()

    # ----------------------------------Construct Optimizer----------------------------------
    print('Construct Optimizer')
    sys.stdout.flush()
    optimizer = None
    if config['model']['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config['model']['learning_rate'],
                                  initial_accumulator_value=1e-8)
    elif config['model']['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    elif config['model']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['model']['learning_rate'])
    elif config['model']['optimizer'] == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=config['model']['learning_rate'], momentum=0.95)

    # ----------------------------------Construct Loss Function----------------------------------
    print('Construct Loss Function')
    sys.stdout.flush()
    criterion = None
    if config['model']['loss_type'] == 'square_loss':
        criterion = nn.MSELoss(reduction='sum')
    elif config['model']['loss_type'] == 'log_loss':  # log_loss
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    # ---------Model Initial Check---------
    if config['model']['evaluation']:
        model.eval()
        print("Model Initial Check: ", end='')
        if config['task'] == 'rating':
            train_result = metrics.RMSE(model, train_loader)
            test_result = metrics.RMSE(model, test_loader)
            if valid_dataset is not None:
                valid_result = metrics.RMSE(model, valid_loader)
                print("Train_RMSE: {:.3f},".format(train_result),
                      "Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(valid_result, test_result))
            else:
                print("Train_RMSE: {:.3f},".format(train_result), "Test_RMSE: {:.3f}".format(test_result))
        elif config['task'] == 'ranking':
            test_hr, test_ndcg = metrics.metrics(model, test_loader)
            test_result = test_hr
            print("Test_HR: {:.3f}, Test_NDCG: {:.3f}".format(test_hr, test_ndcg))
            sys.stdout.flush()

    # ----------------------------------Training----------------------------------
    print('Training...')
    if config['task'] == 'rating':
        best_result = 100
    elif config['task'] == 'ranking':
        best_result = (0, 0)
    else:
        best_result = None

    saved = False
    start_time = time.time()
    for epoch in range(config['model']['epochs']):
        model.train()
        loss = 0  # No effect, ignore this line
        for i, (features, feature_values, label) in enumerate(train_loader):
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
            # ---------checkpoint---------
            if i % config['model']['steps_per_checkpoint'] == 0:
                print(
                    "Running Epoch {:03d}/{:03d}".format(epoch + 1, config['model']['epochs']),
                    "loss:{:.3f}".format(float(loss)),
                    "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))
                start_time = time.time()
                sys.stdout.flush()
                # ----------------------------------Validation----------------------------------
                if config['model']['evaluation']:
                    model.eval()
                    save_flag = False
                    if config['task'] == 'rating':
                        # train_result = metrics.RMSE(model, train_loader)
                        test_result = metrics.RMSE(model, test_loader)
                        if valid_dataset is not None:
                            # valid_result = metrics.RMSE(model, valid_loader)
                            print("\tRunning Epoch {:03d}/{:03d}".format(epoch + 1, config['model']['epochs']),
                                  # "Train_RMSE: {:.3f},".format(train_result),
                                  # "Valid_RMSE: {:.3f},".format(valid_result),
                                  "Test_RMSE: {:.3f}".format(test_result))
                        else:
                            print("\tRunning Epoch {:03d}/{:03d}".format(epoch + 1, config['model']['epochs']),
                                  # "Train_RMSE: {:.3f},".format(train_result),
                                  "Test_RMSE: {:.3f}".format(test_result))

                        save_flag = test_result < best_result
                    elif config['task'] == 'ranking':
                        test_hr, test_ndcg = metrics.metrics(model, test_loader)
                        test_result = (test_hr, test_ndcg)
                        print("\tRunning Epoch {:03d}/{:03d}".format(epoch + 1, config['model']['epochs']),
                              "Test_HR: {:.3f}, Test_NDCG: {:.3f}".format(test_hr, test_ndcg))

                        save_flag = test_hr > best_result[0]
                    else:
                        test_result = best_result  # No effect, ignore this line

                    if save_flag:
                        if config['model']['save']:
                            if 'tag' in config:
                                torch.save(model, os.path.join(config['model']['model_path'],
                                                               '{}_{}.pth'.format(config['model']['name'], config['tag'])))
                            saved = True
                        best_result = test_result

                    sys.stdout.flush()

    # ----------------------------------Evaluation----------------------------------
    if config['model']['evaluation']:
        print('Evaluating...')
        model.eval()
        flag = False
        if config['task'] == 'rating':
            train_result = metrics.RMSE(model, train_loader)
            test_result = metrics.RMSE(model, test_loader)
            if valid_dataset is not None:
                valid_result = metrics.RMSE(model, valid_loader)
                print("Train_RMSE: {:.3f}, Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(train_result, valid_result,
                                                                                         test_result))
            else:
                print("Train_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(train_result, test_result))

            flag = test_result < best_result
        elif config['task'] == 'ranking':
            train_result = metrics.RMSE(model, train_loader)
            test_hr, test_ndcg = metrics.metrics(model, test_loader)
            test_result = (test_hr, test_ndcg)
            print("Train_RMSE: {:.3f}, Test_HR: {:.3f}, Test_NDCG: {:.3f}".format(train_result, test_hr, test_ndcg))

            flag = test_hr > best_result[0]
        else:
            test_result = best_result  # No effect, ignore this line

        if flag:
            best_result = test_result
        print('------Best Result: ', best_result, '------', sep='')
        sys.stdout.flush()

    if not saved and config['model']['save']:
        if 'tag' in config:
            torch.save(model, os.path.join(config['model']['model_path'],
                                           '{}_{}.pth'.format(config['model']['name'], config['tag'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='../config_example/NGCF.yaml',
                        help='path for configure file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("config file doesn't exist")
        exit(-1)
    run(yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader))

