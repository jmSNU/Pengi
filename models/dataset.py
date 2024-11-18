import torch
import pandas as pd
import os
import librosa
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple
import numpy as np
import random

class MediaContentDataset(Dataset):
    def __init__(self, csv_file: str, data_dir: str, train: bool, duration : int, transform: Optional[Callable] = None, sample_rate: int = 44100):
        """
        Args:
            csv_file (str): Path to the CSV file containing the dataset information.
            data_dir (str): Directory with all the .wav files.
            train (bool): Flag to indicate if it's training mode.
            duration(int) : How long the train audio is
            transform (Optional[Callable]): Optional transform to be applied on a sample.
            sample_rate (int): Sample rate for loading the audio files (default: 44100).
        """
        self.sample_files = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.is_train = train
        self.duration = duration
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Load the input audio file
        data_path = os.path.join(self.data_dir, self.sample_files.iloc[index, 0])
        sample, _ = librosa.load(data_path, sr=self.sample_rate, mono=True)
        sample = torch.tensor(sample, dtype=torch.float32)

        if self.duration*self.sample_rate >= sample.shape[0]:
            repeat_factor = int(np.ceil(
                (self.duratino*self.sample_rate)/sample.shape[0]
            ))
            sample = sample.repeat(repeat_factor)
            sample = sample[0:self.duration*self.sample_rate]
        else:
            start_idx = random.randrange(
                sample.shape[0] - self.duration*self.sample_rate
            )
            sample = sample[start_idx:start_idx + self.duration*self.sample_rate]

        # Load the label audio files
        label_paths = [os.path.join(self.data_dir, self.sample_files.iloc[index, i + 1]) for i in range(4)]
        labels = []
        for path in label_paths:
            label, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            label = torch.tensor(label, dtype=torch.float32)
            
            if self.duration*self.sample_rate >= label.shape[0]:
                repeat_factor = int(np.ceil(
                    (self.duratino*self.sample_rate)/label.shape[0]
                ))
                label = label.repeat(repeat_factor)
                label = label[0:self.duration*self.sample_rate]
            else:
                label = label[start_idx:start_idx + self.duration*self.sample_rate]

            labels.append(label)

        # Apply transform if specified
        if self.transform is not None:
            sample = self.transform(sample)
            labels = [self.transform(label) for label in labels]

        return sample, labels
