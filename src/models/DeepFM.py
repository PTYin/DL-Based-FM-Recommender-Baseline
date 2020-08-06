import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, feature_size, field_size,
                 embedding_size, dropout_fm, deep_layers_dim, dropout_deep, act_function, batch_norm
                 ):
        super(DeepFM, self).__init__()

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.dropout_fm = dropout_fm
        self.deep_layers_dim = deep_layers_dim
        self.dropout_deep = dropout_deep
        self.act_function = act_function
        self.batch_norm = batch_norm

        self.embeddings = nn.Embedding(self.feature_size, self.embedding_size)
        self.biases = nn.Embedding(self.feature_size, 1)

        self.dropout_fm_layers = [nn.Dropout(dropout_fm[0]), nn.Dropout(dropout_fm[1])]
        self.dropout_deep_layers = [nn.Dropout(dropout_deep[0]),
                                    nn.Dropout(dropout_deep[1]),
                                    nn.Dropout(dropout_deep[1])]
        self.weight_list = []

        # deep layers
        mlp_module = []
        in_dim = self.field_size * self.embedding_size
        mlp_module.append(self.dropout_deep_layers[0])
        for i in range(len(self.deep_layers_dim)):
            out_dim = self.deep_layers_dim[i]
            mlp_module.append(nn.Linear(in_dim, out_dim))
            self.weight_list.append(mlp_module[-1].weight)
            in_dim = out_dim
            if self.batch_norm:
                mlp_module.append(nn.BatchNorm1d(out_dim))

            if self.act_function == 'relu':
                mlp_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                mlp_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                mlp_module.append(nn.Tanh())

            mlp_module.append(self.dropout_deep_layers[i+1])

        self.deep_layers = nn.Sequential(*mlp_module)
        self.predict_layer = nn.Linear(in_dim+2, 1, bias=True)
        self.weight_list.append(self.predict_layer.weight)

        self._init_weight_()

    def _init_weight_(self):
        # embeddings
        nn.init.normal_(self.embeddings.weight, 0.0, 0.01)
        nn.init.uniform_(self.biases.weight, 0.0, 1.0)

        # deep layers
        for m in self.deep_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.predict_layer.weight)

    def forward(self, features: torch.LongTensor, feature_values: torch.FloatTensor):
        # -------------Embedding-------------
        feature_embeddings = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=2)
        feature_embeddings = feature_embeddings * feature_values

        # -------------FM Component-------------
        # First Order Term
        feature_bias = self.biases(features)
        first_order_bias = (feature_bias * feature_values).sum(dim=1)  # 0 dimension is batch
        first_order_bias = self.dropout_fm_layers[0](first_order_bias)

        # Second Order Term
        sum_square_embed = feature_embeddings.sum(dim=1).pow(2)
        square_sum_embed = (feature_embeddings.pow(2)).sum(dim=1)
        second_order_bias = (0.5 * (sum_square_embed - square_sum_embed)).sum(dim=1).unsqueeze(dim=1)
        second_order_bias = self.dropout_fm_layers[1](second_order_bias)

        # -------------Deep Component-------------

        deep_input = feature_embeddings.view(-1, self.field_size * self.embedding_size)
        y_deep = self.deep_layers(deep_input)

        # -------------Concat-------------
        concat_input = torch.cat((first_order_bias, second_order_bias, y_deep), dim=1)
        out = self.predict_layer(concat_input)
        return out.view(-1)