import torch
import numpy as np
import torch.utils.data as data


class LibFMDataset(data.Dataset):
    """ Construct the FM pytorch dataset. """

    def __init__(self, file, feature_map, node_map=None, user_bought=None):
        super(LibFMDataset, self).__init__()
        self.label = []
        self.features = []
        self.feature_values = []

        with open(file, 'r') as fd:
            line = fd.readline()

            while line:
                items = line.strip().split()

                # convert features
                raw = [item.split(':')[0] for item in items[1:]]
                self.features.append(
                    torch.tensor([feature_map[item] for item in raw], dtype=torch.long))
                self.feature_values.append(torch.tensor(
                    [float(item.split(':')[1]) for item in items[1:]], dtype=torch.float))

                self.label.append(np.float32(items[0]))
                if node_map is not None and user_bought is not None:
                    if self.label[-1] > 0:
                        user_bought[node_map[feature_map[raw[0]]]] = node_map[feature_map[raw[1]]]
                # label = np.float32(1) if float(items[0]) > 0 else np.float32(0)  # positive: 1, negative: 0
                # self.label.append(label)

                line = fd.readline()

        assert all(len(item) == len(self.features[0]
                                    ) for item in self.features), 'features are of different length'

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        features = self.features[idx]
        feature_values = self.feature_values[idx]
        return features, feature_values, label

    @staticmethod
    def read_features(file, features: dict, nodes=None):
        """ Read features from the given file. """
        with open(file, 'r') as fd:
            line = fd.readline()
            while line:
                items = line.strip().split()
                for i, item in enumerate(items[1:]):
                    item = item.split(':')[0]
                    if item not in features:
                        features[item] = len(features)
                        if nodes is not None and (i == 0 or i == 1):
                            nodes[features[item]] = len(nodes)
                line = fd.readline()
