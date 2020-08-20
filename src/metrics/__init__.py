import numpy as np
import torch


def metrics(model, test_loader, top_k=10):
    hr, ndcg = [], []

    for features, feature_values, label in test_loader:
        features = features.cuda()
        feature_values = feature_values.cuda()

        predictions = model(features, feature_values)
        _, indices = torch.topk(predictions, top_k)
        # recommends = torch.take(
        # 		item, indices).cpu().numpy().tolist()

        recommends = indices.cpu().numpy().tolist()

        if 0 in recommends:
            # HR
            hr.append(1)
            # NDCG (if label==1 rel=1, else rel=0; IDCG=1)
            ndcg.append(1 / np.log2(recommends.index(0) + 2))
        else:
            hr.append(0)
            ndcg.append(0)

    return np.mean(hr), np.mean(ndcg)


def RMSE(model, data_loader):
    RMSE = np.array([], dtype=np.float32)
    for features, feature_values, label in data_loader:
        features = features.cuda()
        feature_values = feature_values.cuda()
        label = label.cuda()

        prediction = model(features, feature_values)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())
