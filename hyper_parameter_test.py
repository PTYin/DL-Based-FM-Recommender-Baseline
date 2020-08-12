import yaml
import os


def generate_yaml_doc(config, file):
    file = open(file, 'w', encoding='utf-8')
    yaml.dump(config, file)
    file.close()


if __name__ == '__main__':
    config_map = {'DeepFM': yaml.load(open('config_example/DeepFM.yaml', 'r').read(), Loader=yaml.FullLoader),
                  'XDeepFM': yaml.load(open('config_example/XDeepFM.yaml', 'r').read(), Loader=yaml.FullLoader),
                  'NGCF': yaml.load(open('config_example/NGCF.yaml', 'r').read(), Loader=yaml.FullLoader)}

    datasets = ['automotive', 'books', 'clothing', 'ml-1m', 'office', 'ticket']
    data_base_dir = '/home/share/yinxiangkun/libfm/data/'
    rating_path = dict(zip(datasets, map(lambda dataset: os.path.join(data_base_dir, 'rating/', dataset), datasets)))
    ranking_path = dict(zip(datasets, map(lambda dataset: os.path.join(data_base_dir, 'ranking/', dataset), datasets)))
    for config in config_map.values():
        config['dataset']['batch_size'] = 100
        dataset = datasets[0]
        for key in config['dataset']['paths']:
            config['dataset']['paths'][key] = os.path.join(rating_path[dataset], '{}.{}.libfm'.format(dataset, key))

    # -----------Create Directories-----------
        if not os.path.exists('config/DeepFM/'):
            os.makedirs('config/DeepFM/')
        if not os.path.exists('config/XDeepFM/'):
            os.makedirs('config/XDeepFM/')
        if not os.path.exists('config/NGCF/'):
            os.makedirs('config/NGCF/')

    # -----------DeepFM Hyper Parameter-----------
    config = config_map['DeepFM']
    hyper_params = config['model']['hyper_params']
    dropout = 0.5
    for neurons in [32, 64, 128, 256, 512]:
        for layers in [1, 2, 3, 4]:
            dropout_deep = [dropout] * layers
            deep_layers = [neurons] * layers
            assert 'dropout_deep' in hyper_params and 'deep_layers' in hyper_params
            hyper_params['dropout_deep'] = dropout_deep
            hyper_params['deep_layers'] = deep_layers
            config['tag'] = 'neurons_{}_layers_{}'.format(neurons, layers)
            generate_yaml_doc(config,
                              os.path.join('config/DeepFM/', 'neurons_{}_layers_{}.yaml'.format(neurons, layers)))
