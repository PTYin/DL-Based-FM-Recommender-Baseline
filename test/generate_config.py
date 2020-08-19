import os

DeepFM_template = '''tag: '{dataset}_{embedding_size}'
dataset:
  decoder: 'libfm'
  num_workers: 4
  batch_size: 100  # batch size for training
  bipartite: False
  paths:
    train: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.train.libfm'
    valid: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.valid.libfm'
    test: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.test.libfm'

model:
  name: 'DeepFM'
  loss_type: 'square_loss'  # ['square_loss', 'log_loss']
  optimizer: 'Adam'  # ['Adagrad', 'Adam', 'SGD', 'Momentum']
  learning_rate: 0.001
  epochs: 30
  steps_per_checkpoint: 500
  save: True  # save model or not
  model_path: '/home/share/yinxiangkun/saved/{task}/DeepFM/'
  gpu: '2'  # gpu ID
  load: False
  evaluation: True

  hyper_params:
    activation_function: 'relu'  # ['relu', 'sigmoid', 'tanh', 'identity']
    batch_norm: True  # use batch_norm or not
    embedding_size: {embedding_size}
    dropout_fm: [0.0, 0.0]  # dropout rate for FM
    dropout_deep: [0.0, 0.0]  # dropout rate for MLP
    deep_layers: [128, 128]  # size of layers in MLP model
    lambda: 0.001  # regularizer for deep layers and prediction layers

task: '{task}'  # ['rating', 'ranking']
'''
XDeepFM_template = '''tag: '{dataset}_{embedding_size}'
dataset:
  decoder: 'libfm'
  num_workers: 4
  batch_size: 100  # batch size for training
  bipartite: False
  paths:
    train: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.train.libfm'
    valid: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.valid.libfm'
    test: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.test.libfm'

model:
  name: 'XDeepFM'
  loss_type: 'square_loss'  # ['square_loss', 'log_loss']
  optimizer: 'Adam'  # ['Adagrad', 'Adam', 'SGD', 'Momentum']
  learning_rate: 0.001
  epochs: 30
  steps_per_checkpoint: 500
  save: True  # save model or not
  model_path: '/home/share/yinxiangkun/saved/{task}/XDeepFM/'
  gpu: '2'  # gpu ID
  load: False
  evaluation: True

  hyper_params:
    deep_act: 'relu'  # ['relu', 'sigmoid', 'tanh']
    cin_act: 'relu'  # ['relu', 'sigmoid', 'tanh']
    batch_norm: True  # use batch_norm or not
    cin_split_half: False
    embedding_size: {embedding_size}
    deep_layers: [128, 128]  # size of layers in MLP model
    cin_layers: [256]
    dropout_deep: [0.0, 0.0]
    lambda: 0.001  # regularizer for deep layers and prediction layers

task: '{task}'  # ['rating', 'ranking']
'''
NGCF_template = '''tag: '{dataset}_{embedding_size}'
dataset:
  decoder: 'libfm'
  num_workers: 4
  batch_size: 100  # batch size for training
  bipartite: True  # recommendation in bipartite graph
  paths:
    train: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.train.libfm'
    test: '/home/share/yinxiangkun/libfm/data/{task}/{dataset}/{dataset}.test.libfm'

model:
  name: 'NGCF'
  loss_type: 'square_loss'
  optimizer: 'Adam'  # ['Adagrad', 'Adam', 'SGD', 'Momentum']
  learning_rate: 0.001
  epochs: 30
  steps_per_checkpoint: 500
  save: True  # save model or not
  model_path: '/home/share/yinxiangkun/saved/{task}/NGCF/'
  gpu: '0'  # gpu ID
  load: False
  evaluation: True

  hyper_params:
    activation_function: 'leakyrelu'  # ['relu', 'sigmoid', 'tanh', 'leakyrelu']
    normalized: True
    dropout: [0.1, 0.1, 0.1]  # dropout rate for each GNN layer
    embedding_size: 64  # predictive factors numbers in the model
    layers: [128, 96, 64]  # size of layers in GNN layers
    lambda: 0.0  # regularizer for bilinear layers

task: '{task}'  # ['rating', 'ranking']
'''

config_dir = os.path.join('..', 'config_test')


def generate(task, model, dataset, embedding_size):
    if not os.path.exists(os.path.join(config_dir, task, embedding_size, dataset)):
        os.makedirs(os.path.join(config_dir, task, embedding_size, dataset))

    if model == 'DeepFM':
        config = DeepFM_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    elif model == 'XDeepFM':
        config = XDeepFM_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    elif model == 'NGCF':
        config = NGCF_template.format(task=task, dataset=dataset, embedding_size=embedding_size)
    else:
        raise NotImplementedError('model not in the list')
    open(os.path.join(config_dir, task, embedding_size, dataset, model+'.yaml'), 'w').write(config)


if __name__ == '__main__':
    dataset_list = ["automotive", "books", "clothing", "ml-1m", "office", "ticket"]
    # embedding_size_list = [4, 8, 16, 32, 64, 128, 256, 512]
    embedding_size_list = [256, 512]
    embedding_size_list = list(map(lambda x: str(x), embedding_size_list))
    # ---------------rating---------------
    for dataset in dataset_list:
        for embedding_size in embedding_size_list:
            generate('rating', 'DeepFM', dataset, embedding_size)
            generate('rating', 'XDeepFM', dataset, embedding_size)

    # ---------------ranking---------------
    for dataset in dataset_list:
        for embedding_size in embedding_size_list:
            generate('ranking', 'DeepFM', dataset, embedding_size)
            generate('ranking', 'XDeepFM', dataset, embedding_size)
            generate('ranking', 'NGCF', dataset, embedding_size)
