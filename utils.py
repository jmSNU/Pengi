import os
import argparse
import yaml
import torchaudio
import torchaudio.transforms as T
import numpy as np
import random
import torch

def read_config_as_args(config_path):
    return_dict = {}
    with open(config_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        return_dict[k] = v
    return argparse.Namespace(**return_dict)

def get_latest_checkpoint(model_name):
    base_path = f"./checkpoint/{model_name}"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No checkpoint directory found for model: {model_name}")

    checkpoint_dirs = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))],
        reverse=True
    )

    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found.")

    latest_checkpoint_dir = os.path.join(base_path, checkpoint_dirs[0])

    checkpoint_files = sorted(
        [f for f in os.listdir(latest_checkpoint_dir) if f.startswith("model-epoch-")],
        key=lambda x: int(x.split("-epoch-")[1].split(".")[0]),
        reverse=True
    )

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the latest directory.")

    latest_checkpoint_path = os.path.join(latest_checkpoint_dir, checkpoint_files[0])

    return latest_checkpoint_path

def load_audio_into_tensor(audio_path, audio_duration, sampling_rate, resample=True):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = sampling_rate
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)
        sample_rate = resample_rate

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)