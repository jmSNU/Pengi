import os
import sys
from torch.utils.data import DataLoader
import soundfile as sf

data_dir = "./dataset"
duration = 30
batch_size = 1

if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ))
        from models.dataset import MediaContentDataset
    else:
        from ..models.dataset import MediaContentDataset


    csv_file_path = os.path.join(data_dir, "processed_data", "train.csv")
    dataset = MediaContentDataset(
        csv_file=csv_file_path,
        data_dir=data_dir,
        train=True,
        duration = duration
    )

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    label_name = ["vocals","background","speech", "others"]
    input, labels = next(iter(dl))

    sf.write("./test/input.wav", input.squeeze(0).numpy(), samplerate=44100)
    for idx,label in enumerate(labels):
        sf.write(f"./test/{label_name[idx]}.wav", label.squeeze(0).numpy(), samplerate=44100)
