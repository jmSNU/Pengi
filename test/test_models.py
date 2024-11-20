import sys
import os

if __name__ == "__main__":
    if __package__ is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ))
        from wrapper import MSSWrapper
        from utils import *
    else:
        from ..wrapper import MSSWrapper
        from ..utils import *

    train_config_path = "configs/train.yml"
    base_config_path = "configs/base_mss.yml"

    train_args = read_config_as_args(train_config_path)
    base_args = read_config_as_args(base_config_path)
    merged_dicts = {**vars(base_args), **vars(train_args)}
    args = argparse.Namespace(**merged_dicts)

    mss_wrapper = MSSWrapper(config="train", use_cuda=True)
    model, tokenizer = mss_wrapper.model, mss_wrapper.enc_tokenizer
    print(model)