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
from torchlibrosa.stft import Spectrogram


class SpectralL1Loss(nn.Module):
    def __init__(self, n_fft = 512, hop_size = 256, win_size = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_size
        self.win_length = win_size
        self.spectrogram_extractor = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, 
            win_length=self.win_length, window="hann", center=True, pad_mode="reflect", 
            freeze_parameters=True).to(train_args.device)
        
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted, target):
        loss = 0
        for stem in range(4):
            pred_spectrogram = self.spectrogram_extractor(predicted[:, stem, :])
            target_spectrogram = self.spectrogram_extractor(target[:, stem, :])

            loss += self.l1_loss(pred_spectrogram, target_spectrogram)
        return loss/4

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

        self.loss_fn = SpectralL1Loss(eval_args.n_fft, eval_args.hop_size, eval_args.window_size)


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

    train_args = read_config_as_args(eval_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="eval", use_cuda=True)
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer

    prompts = [PROMPT_VER2]

    evaluator = MSSEvaluator(model, preprocess_text=mss_wrapper.preprocess_text, prompts=prompts, eval_args=args)

    evaluator.evaluate()
