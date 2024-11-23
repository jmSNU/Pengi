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


def visualize_spectrograms(pred_spectrogram, target_spectrogram, stem_index, save_path="spectrogram_comparison.png"):
    # Convert tensors to numpy arrays if they are still in torch format
    if hasattr(pred_spectrogram, "detach"):
        pred_spectrogram = pred_spectrogram.detach().cpu().numpy()
    if hasattr(target_spectrogram, "detach"):
        target_spectrogram = target_spectrogram.detach().cpu().numpy()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Spectrogram Comparison for Stem {stem_index}", fontsize=16)

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
    plt.savefig(save_path)
    plt.close(fig)

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

        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=eval_args.dataset_config["sampling_rate"],
            perceptual_weighting=True,
        )

    def evaluate(self):
        self.model.eval()
        eval_losses = []
        bar = tqdm(total=len(self.eval_dl), desc="Evaluating")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.eval_dl):
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                labels = torch.stack(labels, dim=1)

                text_encs = self.preprocess_text(self.prompts * len(inputs), enc_tok=True, add_text=False)

                outputs = self.model(inputs, text_encs)

                loss = self.loss_fn(outputs, labels)
                eval_losses.append(loss.item())

                bar.update(1)
                bar.set_description(f"[INFO] Eval Loss: {sum(eval_losses) / len(eval_losses):.5f}")
                
                if batch_idx ==0 :
                    plt.figure()
                    plt.plot(outputs[batch_idx, 0, :].cpu().detach().numpy(), label = "prediected")
                    plt.plot(labels[batch_idx, 0].cpu().detach().numpy(), label = "ground truth")
                    plt.legend()
                    plt.savefig("./output.png")

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
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER2]

    evaluator = MSSEvaluator(model, preprocess_text=mss_wrapper.preprocess_text, prompts=prompts, eval_args=args)

    evaluator.evaluate()
