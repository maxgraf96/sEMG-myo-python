import os.path
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import pywt
import scipy.signal as signal
import torch.utils.data
from constants import DATALOADER_NUM_WORKERS
from hyperparameters import BATCH_SIZE, DATA_LEN, DATASET_SHIFT_SIZE, SEQ_LEN, LOSS_INDICES, DATASET_UPSCALE_FACTOR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

data_seq_len = SEQ_LEN - 1


def power_spectral_density(emg_data, fs=1000):
    psd = []
    for channel_data in emg_data.T:
        f, pxx = signal.welch(channel_data, fs, nperseg=50)
        psd.append(pxx)
    return np.array(psd)

def median_frequency(psd, f):
    mdf = []
    for channel_psd in psd:
        cumsum = np.cumsum(channel_psd)
        median_idx = np.where(cumsum >= cumsum[-1] / 2)[0][0]
        mdf.append(f[median_idx])
    return np.array(mdf)

def mean_frequency(psd, f):
    mnf = []
    for channel_psd in psd:
        mean_freq = np.sum(channel_psd * f) / np.sum(channel_psd)
        mnf.append(mean_freq)
    return np.array(mnf)

def peak_frequency(psd, f):
    pf = []
    for channel_psd in psd:
        max_idx = np.argmax(channel_psd)
        pf.append(f[max_idx])
    return np.array(pf)

def extract_frequency_features(emg_data, fs=1000):
    psd = power_spectral_density(emg_data, fs)
    f = np.linspace(0, fs / 2, len(psd[0]))
    
    mdf = median_frequency(psd, f)
    mnf = mean_frequency(psd, f)
    pf = peak_frequency(psd, f)

    return mdf, mnf, pf


class SEMGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "data"):
        super().__init__()

        # Read all csvs in data directory
        data = []
        for filename in os.listdir("data"):
            if filename.endswith(".csv"):
                data.append(pd.read_csv("data" + os.path.sep + filename, index_col=False))
        # Load data from csv
        data_np = []
        for i in range(len(data)):
            data_np.append(data[i].to_numpy(dtype=np.float32))

        self.data = np.concatenate(data_np, axis=0)

        self.train_percentage = 0.95

        # Bandpass self.data from 10-25 Hz
        fs = 50  # Sampling frequency
        fmin = 20  # Minimum frequency to pass
        fmax = 25  # Maximum frequency to pass
        nyq = 0.5 * fs  # Nyquist frequency

        raw_samples = self.data[:, :8]
        labels = self.data[:, 8:-3]
        wrist_angles = self.data[:, -3:]

        # Define filter parameters
        # self.b, self.a = signal.butter(4, [fmin / nyq, fmax / nyq], btype='band')

        self.samples = []
        self.labels = []
        self.wrist_angles = []
        total_num_samples = self.data.shape[0]
        for start in tqdm(range(0, int(total_num_samples * 0.1) - DATA_LEN, DATASET_SHIFT_SIZE)):
        # for start in tqdm(range(0, total_num_samples - DATA_LEN, DATASET_SHIFT_SIZE)):
            # Optional data augmentation
            for aug in range(DATASET_UPSCALE_FACTOR):
                sample = raw_samples[start : start + DATA_LEN].copy()
                label = labels[start : start + DATA_LEN]
                wrist_angle = wrist_angles[start : start + DATA_LEN]

                if aug > 0:
                    std = aug * 0.5
                    # Augment data by adding noise to the sample
                    sample = sample + np.random.normal(0, std, sample.shape)

                sample = self.preprocess_sample(sample, wrist_angle)

                # Append to list
                self.samples.append(sample)
                self.labels.append(label)
                self.wrist_angles.append(wrist_angle)

        print("---")
        print("Dataset length: ", self.__len__())
        print("---")

    def preprocess_sample(self, sample, wrist_angles):
        # =============================================================================
        # Preprocessing
        # =============================================================================
        # Apply filter to data along the first axis
        # sample = signal.filtfilt(self.b, self.a, sample, axis=0)

        coefficients_level4 = pywt.wavedec(sample, 'db2', 'smooth', level=4, axis=0)
        cA, cD = coefficients_level4[0], coefficients_level4[1]
        # Concat cA and cD
        wavelet_features = np.concatenate((cA, cD), axis=0)

        # Get features
        # Time domain features
        # Mean absolute value
        mean_abs = np.mean(np.abs(sample), axis=0).reshape(1, -1)
        # Root mean square
        rms = np.sqrt(np.mean(np.square(sample), axis=0)).reshape(1, -1)
        # Variance
        var = np.var(sample, axis=0).reshape(1, -1)

        mdf, mnf, pf = extract_frequency_features(sample, fs=50)
        mdf = mdf.reshape(1, -1)
        mnf = mnf.reshape(1, -1)
        pf = pf.reshape(1, -1)

        # Make array of zeros for wrist angle
        wrist_angle = np.zeros(sample.shape)
        wrist_angle[:, :3] = wrist_angles

        # Prepend features to sample
        sample = np.concatenate((mean_abs, rms, var, mdf, mnf, pf, wavelet_features, sample, wrist_angle), axis=0)

        # replace all nan values with 0
        sample = np.nan_to_num(sample, copy=False)

        # Convert to float32
        sample = sample.astype(np.float32)

        return sample

    def __getitem__(self, index):
        sample = torch.tensor(self.samples[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return {'sample': sample, 'label': label, 'wrist_angle': self.wrist_angles[index]}

    def __len__(self):
        return len(self.samples) * DATASET_UPSCALE_FACTOR - data_seq_len


class SEMGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = SEMGDataset(self.data_dir)
        self.train = None
        self.val = None

    def setup(self, stage: str):
        # Split dataset into train and validation
        train_size = int(self.dataset.train_percentage * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train, self.val = random_split(self.dataset, [train_size, val_size])
        # Non-random split
        # self.train = torch.utils.data.Subset(self.dataset, range(train_size))
        # self.val = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=False, drop_last=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)
    

if __name__ == '__main__':
    dm = SEMGDataModule(data_dir="data")
    dm.setup(stage="fit")

    # Get a sample from the dataset
    item = dm.train[0]

