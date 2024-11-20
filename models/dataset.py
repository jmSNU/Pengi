import torch
import pandas as pd
import os
import librosa
from torch.utils.data import Dataset
from typing import Optional, Callable, List, Tuple
import numpy as np
import random
from utils import *

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
        sample = load_audio_into_tensor(data_path, self.duration, self.sample_rate, resample=True)

        # Load the label audio files
        label_paths = [os.path.join(self.data_dir, self.sample_files.iloc[index, i + 1]) for i in range(4)]
        labels = []
        for path in label_paths:
            label = load_audio_into_tensor(path, self.duration, self.sample_rate, resample=True)
            labels.append(label)

        # Apply transform if specified
        if self.transform is not None:
            sample = self.transform(sample)
            labels = [self.transform(label) for label in labels]

        return sample, labels
