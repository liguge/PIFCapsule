import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024




def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


# --------------------------------------------------------------------------------------------------------------------
class G(object):
    num_classes = 12
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        data = np.load(self.data_dir, allow_pickle=True).item()
        data, lab = data['data'], data['labels']

        list_data = [data.tolist(), lab.tolist()]
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=(1-(10/2000)), random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            print(len(train_dataset))
            print(len(val_dataset))
            return train_dataset, val_dataset

if __name__ == "__main__":
    path = 'E:\he_fault_diagnosis\paper5\DL-based-Intelligent-Diagnosis-Benchmark-master\DataPaper5\G\\G_200.npy'
    model = G(data_dir=path, normlizetype="0-1").data_preprare()
    print("done!")