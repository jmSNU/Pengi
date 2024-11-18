from wrapper import MSSWrapper
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse
import os
from models.dataset import MediaContentDataset
from transformers import get_linear_schedule_with_warmup
from prompts import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class MSSTrainer:
    def __init__(self, model, preprocess_text, prompts, train_args):
        self.device = torch.device(train_args.device)
        self.model = model.to(self.device)
        self.model_name = train_args.model_name
        self.learning_rate = train_args.learning_rate
        self.weight_decay = train_args.weight_decay
        self.data_dir = train_args.dataset_dir
        self.batch_size = train_args.batch_size
        self.num_epochs = train_args.num_epochs
        self.warmup_steps_ratio = train_args.warmup_steps_ratio
        self.grad_accum_steps = train_args.grad_accum_steps
        self.gradient_clipping = train_args.gradient_clipping
        self.duration = train_args.dataset_config["duration"]
        self.save_checkpoint_epoch = train_args.save_checkpoint_epoch
        self.preprocess_text = preprocess_text
        self.prompts = prompts
        self.save_dir_path = f"./checkpoint/{self.model_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.writer = SummaryWriter(log_dir=self.save_dir_path)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        csv_file_path = os.path.join(self.data_dir, "processed_data", "train.csv")
        self.dataset = MediaContentDataset(
            csv_file=csv_file_path,
            data_dir=self.data_dir,
            train=True,
            duration = self.duration
        )

        self.train_dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        total_training_steps = len(self.train_dl) * self.num_epochs
        warmup_steps = int(total_training_steps * self.warmup_steps_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        self.loss_fn = nn.L1Loss()

    def train(self):
        self.model.train()
        print(f"Starting training for {self.num_epochs} epochs")

        global_step = 0
        
        for epoch in range(self.num_epochs):
            epoch_losses = []
            os.makedirs(self.save_dir_path, exist_ok=True)

            bar = tqdm(total=len(self.train_dl), desc=f"Epoch {epoch}")
            for batch_idx, (inputs, labels) in enumerate(self.train_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]

                text_encs = self.preprocess_text(prompts*len(inputs), enc_tok = True, add_text = False)

                outputs = self.model(inputs, text_encs)
                loss = self.loss_fn(outputs, torch.stack(labels, dim=1))

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.writer.add_scalar("Loss/train", loss.item(), global_step)
                epoch_losses.append(loss.item())
                global_step += 1

                epoch_losses.append(loss.item())
                bar.update(1)
                bar.set_description(f"Epoch {epoch} loss: {sum(epoch_losses) / len(epoch_losses):.5f}")


            if epoch % self.save_checkpoint_epoch == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir_path, f"model-epoch-{epoch}.pt"))
                print(f"\nEpoch[{epoch}] ended with loss: [{sum(epoch_losses) / len(epoch_losses):.5f}]")
                self.writer.add_scalar("Loss/epoch", sum(epoch_losses) / len(epoch_losses), epoch)

        self.writer.close()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def read_config_as_args(config_path):
        return_dict = {}
        with open(config_path, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            return_dict[k] = v
        return argparse.Namespace(**return_dict)

    train_config_path = "configs/train.yml"
    base_config_path = "configs/base.yml"

    train_args = read_config_as_args(train_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="base", use_cuda=True)
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER1]

    trainer = MSSTrainer(model, preprocess_text=mss_wrapper.preprocess_text, prompts=prompts, train_args = args)
    trainer.train()
