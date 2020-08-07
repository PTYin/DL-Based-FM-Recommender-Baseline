import torch
from torch import nn
from layers import CIN
from layers import MLP


class XDeepFM(nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, deep_layers_dim, cin_layers_size, cin_split_half,
                 dropout_deep,
                 deep_act, cin_act, batch_norm, l2):
        super(XDeepFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers_dim = deep_layers_dim
        self.cin_layers_size = cin_layers_size
        self.cin_split_half = cin_split_half
        self.dropout_deep = dropout_deep
        self.l2 = l2
        self.deep_act = deep_act
        self.cin_act = cin_act
        self.batch_norm = batch_norm

        self.embeddings = nn.Embedding(self.feature_size, self.embedding_size)
        self.biases = nn.Embedding(self.feature_size, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        self.weight_list = []

        # deep layers
        in_dim = self.field_size * self.embedding_size
        self.deep_layers = MLP(in_dim, self.deep_layers_dim, self.dropout_deep, self.deep_act, self.batch_norm)
        self.deep_linear = nn.Linear(self.deep_layers_dim[-1], 1, bias=False)

        # CIN
        if cin_split_half:
            self.feature_map_num = sum(
                cin_layers_size[:-1]) // 2 + cin_layers_size[-1]
        else:
            self.feature_map_num = sum(cin_layers_size)
        self.cin = CIN(self.field_size, self.cin_layers_size, self.cin_act, cin_split_half)
        self.cin_linear = nn.Linear(self.feature_map_num, 1, bias=False)

        # Construct weight list
        self.weight_list.append(self.biases.weight)
        self.weight_list += self.deep_layers.weight_list
        self.weight_list.append(self.deep_linear.weight)
        self.weight_list += self.cin.weight_list
        self.weight_list.append(self.cin_linear.weight)

        self._init_weight_()

    def _init_weight_(self):
        # embeddings
        nn.init.normal_(self.embeddings.weight, 0.0, 0.01)
        nn.init.uniform_(self.biases.weight, 0.0, 1.0)

        # deep layers
        for m in self.deep_layers.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.deep_linear.weight)

        # CIN
        for m in self.cin.conv1ds:
            nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(self.cin_linear.weight)

    def forward(self, features: torch.LongTensor, feature_values: torch.FloatTensor):
        # -------------Embedding-------------
        feature_embeddings = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=2)
        feature_embeddings = feature_embeddings * feature_values

        # -------------Linear Component-------------
        feature_bias = self.biases(features)
        linear_out = (feature_bias * feature_values).sum(dim=1)  # 0 dimension is batch

        # -------------Deep Component-------------
        deep_in = feature_embeddings.view(-1, self.field_size * self.embedding_size)
        deep_out = self.deep_layers(deep_in)
        deep_out = self.deep_linear(deep_out)

        # -------------CIN Component-------------
        cin_in = feature_embeddings
        cin_out = self.cin(cin_in)
        cin_out = self.cin_linear(cin_out)

        # -------------Prediction-------------
        return torch.sigmoid(linear_out+deep_out+cin_out+self.global_bias).view(-1)

    def l2_regularization(self):
        l2_reg = 0
        for weight in self.weight_list:
            l2_reg += weight.norm()
        return self.l2 * l2_reg

