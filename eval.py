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
from utils import *
import matplotlib.pyplot as plt
import auraloss

label_names = ["vocals","background","speech", "others"]

def visualize_spectrograms(pred_signal, target_signal, stem_index, eval_dir,
                           n_fft=1024, hop_length=512, win_length=1024, window=torch.hann_window):
    # Ensure signals are 2D for batch processing (batch_size=1 if 1D)
    if pred_signal.dim() == 1:
        pred_signal = pred_signal.unsqueeze(0)
    if target_signal.dim() == 1:
        target_signal = target_signal.unsqueeze(0)

    # Create the window
    window_fn = window(win_length).to(pred_signal.device)

    # Compute STFT
    pred_spectrogram = torch.stft(pred_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                                  window=window_fn, return_complex=True)
    target_spectrogram = torch.stft(target_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                                    window=window_fn, return_complex=True)

    # Convert to magnitude spectrograms
    pred_spectrogram = pred_spectrogram.abs().squeeze(0).cpu().numpy()
    target_spectrogram = target_spectrogram.abs().squeeze(0).cpu().numpy()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Spectrogram Comparison for Stem : {label_names[stem_index]}", fontsize=16)

    # Plot predicted spectrogram
    ax = axes[0]
    img_pred = ax.imshow(pred_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title("Predicted Spectrogram")
    fig.colorbar(img_pred, ax=ax, orientation='vertical')

    # Plot target spectrogram
    ax = axes[1]
    img_target = ax.imshow(target_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title("Target Spectrogram")
    fig.colorbar(img_target, ax=ax, orientation='vertical')

    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"{eval_dir}/spectrogram_{label_names[stem_index]}.png"
    plt.savefig(save_path)
    plt.close(fig)

def visualize_time_signals(pred_signal, target_signal, stem_index, eval_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Time Signal Comparison for Stem : {label_names[stem_index]}", fontsize=16)
    ax = axes[0]
    ax.plot(pred_signal.cpu().detach().numpy())
    ax.set_title("Predicted time signal")

    ax = axes[1]
    ax.plot(target_signal.cpu().detach().numpy(), label = "ground truth")
    ax.set_title("Target time signal")
    plt.savefig(f"{eval_dir}/time_signal_{label_names[stem_index]}.png")       
    plt.close(fig)             

class MSSEvaluator:
    def __init__(self, model, save_path, preprocess_text, prompts, eval_args):
        self.device = torch.device(eval_args.device)
        self.model = model.to(self.device)
        self.model_name = eval_args.model_name
        self.data_dir = eval_args.dataset_dir
        self.batch_size = eval_args.batch_size
        self.duration = eval_args.dataset_config["duration"]
        self.preprocess_text = preprocess_text
        self.prompts = prompts
        self.save_dir = os.path.dirname(save_path)

        csv_file_path = os.path.join(self.data_dir, "processed_data_0.3", "test.csv")
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

        # self.sisdr_loss = auraloss.time.SISDRLoss()
        self.mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=eval_args.dataset_config["sampling_rate"],
            perceptual_weighting=True,
        )

        # self.sisdr_weight = 0.7
        self.mrstft_weight = 1.0

    def evaluate(self):
        self.model.eval()
        eval_losses = []
        bar = tqdm(total=len(self.eval_dl), desc="Evaluating")
        eval_dir = os.path.join(self.save_dir, "eval")
        os.makedirs(name = eval_dir, exist_ok = True)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.eval_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                labels = torch.stack(labels, dim=1)

                text_encs = self.preprocess_text(self.prompts * len(inputs), enc_tok=True, add_text=False)

                outputs = self.model(inputs, text_encs)
                
                total_loss = 0.0
                for stem in range(4):
                    # sisdr_loss = self.sisdr_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    mrstft_loss = self.mrstft_loss(outputs[:, stem, :].unsqueeze(1), labels[:, stem, :].unsqueeze(1))
                    
                    # combined_loss = self.sisdr_weight * sisdr_loss + self.mrstft_weight * mrstft_loss
                    # total_loss += combined_loss
                    total_loss += mrstft_loss
                
                total_loss /= 4
                eval_losses.append(total_loss.item())

                bar.update(1)
                bar.set_description(f"[INFO] Eval Loss: {sum(eval_losses) / len(eval_losses):.5f}")
                
                for stem in range(4):
                    pred_signal = outputs[0, stem, :]
                    gt_signal = labels[0, stem]
                    visualize_spectrograms(
                        pred_signal,
                        gt_signal,
                        stem,
                        eval_dir
                        )

                    visualize_time_signals(
                        pred_signal,
                        gt_signal,
                        stem,
                        eval_dir
                    ) 

        avg_loss = sum(eval_losses) / len(eval_losses)
        print(f"\n[INFO] Evaluation completed with average loss: {avg_loss:.5f}")
        return avg_loss

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    eval_config_path = "configs/demo.yml"
    base_config_path = "configs/base_mss.yml"

    eval_config = read_config_as_args(eval_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(eval_config)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="eval", use_cuda=True)
    save_path = mss_wrapper.model_path
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER1]

    evaluator = MSSEvaluator(
        model, 
        save_path,
        preprocess_text=mss_wrapper.preprocess_text, 
        prompts=prompts, 
        eval_args=args
    )

    evaluator.evaluate()
