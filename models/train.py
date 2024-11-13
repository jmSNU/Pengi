from wrapper import MSS
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse


train_audio_file_paths = ["FILE_PATH_1", "FILE_PATH_2"]
text_prompts = ["generate metadata", "generate metadata"]
train_config_path = "configs/train.yml"
base_config_path = "configs/base.yml"


def read_config_as_args(config_path):
    return_dict = {}
    with open(config_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        return_dict[k] = v
    return argparse.Namespace(**return_dict)

def train():

    # argument parsing
    train_args = read_config_as_args(train_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    device = torch.device(args.device)
    mss_wrapper = MSS(config = "base")
    model, _, args = mss_wrapper.get_model_and_tokenizer()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.learning_rate)

    train_dataset = mss_wrapper.load_audio_into_tensor(train_audio_file_paths, args.duration)

    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader):
            audio, texts_enc, target = batch
            audio, texts_enc, target = audio.to(device), texts_enc.to(device), target.to(device)

            output = model(audio, texts_enc)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), args.save_path)
