# DL-Based-FM-Recommender-Baseline
Several Deep Learning based Factorization Machine Recommendation models implemented by PyTorch

## Dataset

- [MovieLens 25M Dataset](http://files.grouplens.org/datasets/movielens/ml-25m.zip)

We use the same input format as the LibFM toolkit (http://www.libfm.org/).
Split the data to train/test/validation files to run the codes directly.

## Example to Run:
```
python main.py --batch_size=128 --lr=0.05 --hidden_factor=128
```

## Notice

If the parameter num_workers of DataLoader is not equal to 0, please make sure exec the program in Shell environment. 