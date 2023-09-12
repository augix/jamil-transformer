from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 102,
        "dropout": 0.1,
        "num_layers": 3,
        "num_heads": 6,
        "d_model": 36,
        "lang_src": "before_ko",
        "lang_tgt": "ko",
        "model_folder": "weights",
        "model_basename": "L3H6D36maxPos200_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/L3H6D36maxPos200"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


