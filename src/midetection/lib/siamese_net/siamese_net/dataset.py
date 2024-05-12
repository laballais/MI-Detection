import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from midetection.lib.siamese_net.siamese_net import config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# preprocessing and loading the dataset


class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, wallMotion=None, myocardialThickening=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv, header=None)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform
        self.wM = wallMotion
        self.mT = myocardialThickening

    def __getitem__(self, index):
        feature_wm = []
        feature_mt = []

        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        input_index = self.train_df.iat[index, 0].split("_e", 1)[0]

        feature_wm = [
            self.wM[input_index][0], self.wM[input_index][1], self.wM[input_index][
                2], self.wM[input_index][3], self.wM[input_index][4], self.wM[input_index][5]
        ]
        feature_mt = [
            self.mT[input_index][0], self.mT[input_index][1], self.mT[input_index][
                2], self.mT[input_index][3], self.mT[input_index][4], self.mT[input_index][5]
        ]

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(np.array(feature_wm, dtype=np.float32)),
            torch.from_numpy(np.array(feature_mt, dtype=np.float32)),
            torch.from_numpy(
                np.array([1 - int(self.train_df.iat[index, 2])],
                         dtype=np.float32)
            ),
            input_index
        )

    def __len__(self):
        return len(self.train_df)


def get_data_loader(wallMotion, myocardialThickening, data):
    training_csv = config.EchoPaths.training_csv
    training_dir = config.EchoPaths.training_dir
    validating_csv = config.EchoPaths.validating_csv
    validating_dir = config.EchoPaths.validating_dir
    testing_csv = config.EchoPaths.testing_csv
    testing_dir = config.EchoPaths.testing_dir

    if data == "MRI":
        training_csv = config.MRIPaths.training_csv
        training_dir = config.MRIPaths.training_dir
        validating_csv = config.MRIPaths.validating_csv
        validating_dir = config.MRIPaths.validating_dir
        testing_csv = config.MRIPaths.testing_csv
        testing_dir = config.MRIPaths.testing_dir

    # Load the the dataset from raw image folders
    training_dataset = SiameseDataset(
        training_csv,
        training_dir,
        wallMotion,
        myocardialThickening,
        transform=transforms.Compose(
            # , transforms.Normalize((0.5,),(0.5,))
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )

    # Load the dataset as pytorch tensors using dataloader
    train_dataloader = DataLoader(
        training_dataset, shuffle=True, batch_size=config.batch_size
    )

    # Load the the dataset from raw image folders
    validating_dataset = SiameseDataset(
        validating_csv,
        validating_dir,
        wallMotion,
        myocardialThickening,
        transform=transforms.Compose(
            # , transforms.Normalize((0.5,),(0.5,))
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )

    # Load the dataset as pytorch tensors using dataloader
    validating_dataloader = DataLoader(
        validating_dataset, shuffle=True, batch_size=config.batch_size
    )

    # Load the test dataset
    test_dataset = SiameseDataset(
        testing_csv,
        testing_dir,
        wallMotion,
        myocardialThickening,
        transform=transforms.Compose(
            # ,  transforms.Normalize((0.5,),(0.5,))
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, validating_dataloader
