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
    def __init__(self, csv_file: str, data_dir: str, train: bool, duration : int, input_transform: Optional[Callable] = None, label_transform: Optional[Callable] = None, sample_rate: int = 44100):
        self.sample_files = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.is_train = train
        self.duration = duration
        self.input_transform = input_transform
        self.label_transform = label_transform
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
        if self.input_transform is not None:
            sample = self.input_transform(sample)
        if self.label_transform is not None
            labels = [self.label_transform(label) for label in labels]

        return sample, labels
