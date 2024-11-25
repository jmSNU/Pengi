from wrapper import MSSWrapper 
from prompts import *
import os
import soundfile as sf
from  utils import *

def create_demo(audio_names, prompts, eval_config):
    audio_file_paths = [
        f"dataset/processed_data/train/{audio_name}" for audio_name in audio_names
    ]

    mss_wrapper = MSSWrapper(config="eval", use_cuda=True)
    sample_rate = mss_wrapper.args.dataset_config["sampling_rate"]
    checkpoint_path = mss_wrapper.model_path
    date = checkpoint_path.split("/")[-2]

    output = mss_wrapper.predict(
        audio_paths=audio_file_paths,
        text_prompts=prompts,
        audio_resample=True
    )

    save_path = os.path.join(eval_config.save_dir_path, date)
    save_audio(output, audio_names, sample_rate, save_path)

def save_audio(output, audio_names, sample_rate, save_dir):
    label_names = ["vocals","background","speech", "others"]
    num_audio_ids, num_layers, _ = output.shape

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_audio_ids):
        audio_name = audio_names[i]
        audio_dir = os.path.join(save_dir, audio_name)
        os.makedirs(audio_dir, exist_ok=True)

        for layer_idx in range(num_layers):
            audio_data = output[i, layer_idx].cpu().detach().numpy()
            file_path = os.path.join(audio_dir, f"{label_names[layer_idx]}.wav")

            sf.write(file_path, audio_data, samplerate=sample_rate)
            print(f"Saved: {file_path}")

    print("All audio files saved successfully.")

if __name__ == "__main__":

    audio_names = [
        "mix_Music Delta - Disco.wav",
        "mix_The Wrong'Uns - Rothko.wav"
        ]
    
    eval_config_path = "configs/demo.yml"
    eval_config = read_config_as_args(eval_config_path)
    prompts = [PROMPT_VER1]*len(audio_names)
    create_demo(audio_names, prompts, eval_config)