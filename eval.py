from wrapper import MSSWrapper
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import yaml
import argparse
from models.dataset import MediaContentDataset
from tqdm import tqdm
from prompts import *

class MSSEvaluator:
    def __init__(self, model, preprocess_text, prompts, eval_args):
        self.device = torch.device(eval_args.device)
        self.model = model.to(self.device)
        self.model_name = eval_args.model_name
        self.data_dir = eval_args.dataset_dir
        self.batch_size = eval_args.batch_size
        self.duration = eval_args.dataset_config["duration"]
        self.preprocess_text = preprocess_text
        self.prompts = prompts

        self.loss_fn = nn.L1Loss()

        csv_file_path = os.path.join(self.data_dir, "processed_data", "test.csv")
        self.dataset = MediaContentDataset(
            csv_file=csv_file_path,
            data_dir=self.data_dir,
            train=False,
            duration=self.duration,
        )

        self.eval_dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def evaluate(self, checkpoint_path):
        self.model.eval()
        print(f"Starting evaluation using checkpoint: {checkpoint_path}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        eval_losses = []
        bar = tqdm(total=len(self.eval_dl), desc="Evaluating")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.eval_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]

                text_encs = self.preprocess_text(self.prompts * len(inputs), enc_tok=True, add_text=False)

                outputs = self.model(inputs, text_encs)

                loss = self.loss_fn(outputs, torch.stack(labels, dim=1))
                eval_losses.append(loss.item())

                bar.update(1)
                bar.set_description(f"Eval Loss: {sum(eval_losses) / len(eval_losses):.5f}")

        avg_loss = sum(eval_losses) / len(eval_losses)
        print(f"\nEvaluation completed with average loss: {avg_loss:.5f}")
        return avg_loss

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


    train_config_path = "configs/train.yml"
    base_config_path = "configs/base.yml"

    train_args = read_config_as_args(train_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="base", use_cuda=True)
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER1]

    evaluator = MSSEvaluator(model, preprocess_text=mss_wrapper.preprocess_text, prompts=prompts, eval_args=args)

    checkpoint_path = get_latest_checkpoint(train_args.model_name)
    print(f"Loading checkpoint from: {checkpoint_path}")

    evaluator.evaluate(checkpoint_path)
