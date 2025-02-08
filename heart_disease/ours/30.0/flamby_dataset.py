import sys
import os

# file_dir = os.path.dirname("/Users/ashkan/Desktop/PhD/Projects/FL4E-Analysis/")
# sys.path.append(file_dir)

from flamby.datasets.fed_heart_disease import FedHeartDisease

# data_dir = '/raid/home/dgxuser16/NTL/mccarthy/ahmad/github/data/heart'


def FedHeart():
    train_datasets = []
    test_datasets = []

    for center_id in range(4):  # Assuming there are 4 centers in total
        # center_train_dataset = FedHeartDisease(center=center_id, train=True, data_path=data_dir)
        # center_test_dataset = FedHeartDisease(center=center_id, train=False, data_path=data_dir)

        center_train_dataset = FedHeartDisease(center=center_id, train=True)
        center_test_dataset = FedHeartDisease(center=center_id, train=False)

        train_datasets.append(center_train_dataset)
        test_datasets.append(center_test_dataset)

    return train_datasets, test_datasets
