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
from utils import *
from torch.utils.tensorboard import SummaryWriter
import auraloss

class LabelToMaskTransform:
    def __init__(self, eps: float = 1e-8):
        """
        Transform to convert label (source components) to ground truth masks.

        Args:
            eps (float): Small value to avoid division by zero (default: 1e-8).
        """
        self.eps = eps

    def __call__(self, mixed_audio: torch.Tensor, source_components: torch.Tensor):
        """
        Convert labels to ground truth masks.

        Args:
            mixed_audio (torch.Tensor): Mixed audio, shape (T,)
            source_components (torch.Tensor): Source components, shape (4, T)

        Returns:
            torch.Tensor: Ground truth masks, shape (4, T)
        """
        mixed_audio = mixed_audio.unsqueeze(0)  # Shape: (1, T)
        masks = source_components / (mixed_audio + self.eps)  # Shape: (4, T)
        masks = torch.clamp(masks, min=0.0, max=1.0)  # Ensure valid range
        return masks

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
        self.val_step = train_args.val_step
        self.preprocess_text = preprocess_text
        self.prompts = prompts
        self.save_dir_path = f"./checkpoint/{self.model_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.writer = SummaryWriter(log_dir=self.save_dir_path)

        self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        
        self.transform = LabelToMaskTransform()

        csv_file_path = os.path.join(self.data_dir, "processed_data_0.3", "train.csv")
        self.train_dataset = MediaContentDataset(
            csv_file=csv_file_path,
            data_dir=self.data_dir,
            train=True,
            duration = self.duration,
            label_transform=self.transform
        )

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        csv_file_path = os.path.join(self.data_dir, "processed_data_0.3", "validation.csv")
        self.val_dataset = MediaContentDataset(
            csv_file=csv_file_path,
            data_dir=self.data_dir,
            train=True,
            duration = self.duration,
            label_transform=self.transform
        )

        self.val_dl = DataLoader(
            self.train_dataset,
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

        # self.sisdr_loss = auraloss.time.SISDRLoss()
        self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=train_args.dataset_config["sampling_rate"],
            perceptual_weighting=True,
        )

        # self.sisdr_weight = 0.2
        self.mrstft_weight = 1.0

        print(f"[INFO] Number of Trainable Parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"[INFO] Number of Nontrainable Parameters : {sum(p.numel() for p in self.model.parameters() if not p.requires_grad)}")

    def validate(self):
        """
        Run validation step
        """
        self.model.eval()  # Set model to evaluation mode
        val_losses = []

        with torch.no_grad():  # Disable gradient computation
            bar = tqdm(total=len(self.val_dl), desc="Validation")
            for batch_idx, (inputs, labels) in enumerate(self.val_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                labels = torch.stack(labels, dim=1)

                text_encs = self.preprocess_text(prompts * len(inputs), enc_tok=True, add_text=False)

                outputs = self.model(inputs, text_encs)

                total_loss = 0.0
                # total_mrstft_loss = 0.0
                # total_sisdr_loss = 0.0

                for stem in range(4):
                    # sisdr_loss = self.sisdr_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    mrstft_loss = self.mrstft_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    
                    # combined_loss = self.sisdr_weight * sisdr_loss + self.mrstft_weight * mrstft_loss
                    # total_loss += combined_loss
                    total_loss += mrstft_loss
                    # total_mrstft_loss += mrstft_loss.item()
                    # total_sisdr_loss += sisdr_loss.item()                    

                total_loss /= 4
                # total_mrstft_loss /= 4
                # total_sisdr_loss /= 4

                val_losses.append(total_loss.item())
                bar.update(1)
                bar.set_description(f"Validation loss: {sum(val_losses) / len(val_losses):.5f}")

        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"[INFO] Validation Loss: {mean_val_loss:.5f}")
        self.writer.add_scalar("Loss/validation", mean_val_loss, self.global_step)

        return mean_val_loss

    def train(self):
        print(f"[INFO] Starting training for {self.num_epochs} epochs")

        self.global_step = 0

        os.makedirs(self.save_dir_path, exist_ok=True)
        train_args_path = os.path.join(self.save_dir_path, "train_args.yml")
        with open(train_args_path, "w") as f:
            yaml.dump(vars(train_args), f)
        print(f"[INFO] Training arguments saved to {train_args_path}")

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            epoch_losses = []
            self.model.train()
            assert self.model.training

            bar = tqdm(total=len(self.train_dl), desc=f"Epoch {epoch}")
            for batch_idx, (inputs, labels) in enumerate(self.train_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                labels = torch.stack(labels, dim=1)

                text_encs = self.preprocess_text(prompts * len(inputs), enc_tok=True, add_text=False)

                outputs = self.model(inputs, text_encs)

                total_loss = 0.0
                # total_mrstft_loss = 0.0
                # total_sisdr_loss = 0.0

                for stem in range(4):
                    # sisdr_loss = self.sisdr_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    mrstft_loss = self.mrstft_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    total_loss += mrstft_loss
                    # combined_loss = self.sisdr_weight * sisdr_loss + self.mrstft_weight * mrstft_loss
                    # total_loss += combined_loss
                    # total_mrstft_loss += mrstft_loss.item()
                    # total_sisdr_loss += sisdr_loss.item()                    

                total_loss /= 4
                # total_mrstft_loss /= 4
                # total_sisdr_loss /= 4

                loss = total_loss.clone()
                loss /= self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dl):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # self.writer.add_scalar("Loss/SI-SDR", total_sisdr_loss, self.global_step)
                # self.writer.add_scalar("Loss/MR-STFT", total_mrstft_loss, self.global_step)
                self.writer.add_scalar("Loss/train", total_loss.item(), self.global_step)

                epoch_losses.append(total_loss.item())
                self.global_step += 1

                bar.update(1)
                bar.set_description(f"[INFO] Epoch {epoch} loss: {sum(epoch_losses) / len(epoch_losses):.5f}")

            if self.global_step % self.val_step == 0:
                val_loss = self.validate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir_path, "best_model.pt"))
                    print(f"[INFO] Best model saved with validation loss: {best_val_loss:.5f}")

            # Save model checkpoint every few epochs
            if epoch % self.save_checkpoint_epoch == 0:
                 # Validation at the end of each epoch
                
                torch.save(self.model.state_dict(), os.path.join(self.save_dir_path, f"model-epoch-{epoch}.pt"))
                print(f"\n[INFO] Epoch[{epoch}] ended with loss: [{sum(epoch_losses) / len(epoch_losses):.5f}]")
                self.writer.add_scalar("Loss/epoch", sum(epoch_losses) / len(epoch_losses), epoch)

        self.writer.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_config_path = "configs/train.yml"
    base_config_path = "configs/base_mss.yml"

    train_args = read_config_as_args(train_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="train", use_cuda=True)
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER1]

    trainer = MSSTrainer(model, preprocess_text=mss_wrapper.preprocess_text, prompts=prompts, train_args = args)
    trainer.train()
