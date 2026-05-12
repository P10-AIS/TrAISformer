import argparse
import logging
import os
import numpy as np
import torch
from Config.parser import parse_config
from Types.dataset_predictions import DatasetPredictions
from data_handler import load_data, ROI
from dataclasses import dataclass
import yaml
import datasets
import models
import trainers
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================


@dataclass
class TrainConfig:
    name: str = "trAISformer"
    device: str = "cuda:0"
    num_workers: int = 0
    train_split: float = 0.8
    max_epochs: int = 50
    batch_size: int = 32
    token_interval_seconds: int = 600
    init_seqlen: int = 18
    max_seqlen: int = 120
    min_seqlen: int = 36
    lat_size: int = 250
    lon_size: int = 270
    sog_size: int = 30
    cog_size: int = 72
    n_lat_embd: int = 256
    n_lon_embd: int = 256
    n_sog_embd: int = 128
    n_cog_embd: int = 128
    blur: bool = True
    blur_learnable: bool = False
    blur_loss_w: float = 1.0
    blur_n: int = 2
    n_head: int = 8
    n_layer: int = 8
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    grad_norm_clip: float = 1.0
    weight_decay: float = 0.1
    lr_decay: bool = True
    warmup_tokens: float = 375e6
    final_tokens: float = 260e9
    sample_mode: str = "pos_vicinity"
    r_vicinity: int = 40
    top_k: int = 1


@dataclass
class TestConfig:
    track_mlflow: bool = False
    device: str = "cuda:0"
    batch_size: int = 32
    token_interval_seconds: int = 600
    init_seqlen: int = 18
    max_seqlen: int = 120
    min_seqlen: int = 36
    sample_mode: str = "pos_vicinity"
    r_vicinity: int = 40
    top_k: int = 1

# ==========================================
# 2. DATA PREPARATION
# ==========================================


def prepare_train_dataloaders(config: TrainConfig, dataset_path: str):
    device = torch.device(config.device)
    loaded_splits, roi = load_data(
        [config.train_split, 1 -
            config.train_split], dataset_path, config.token_interval_seconds, config.min_seqlen
    )

    print(f"Train: {len(loaded_splits[0])}, Val: {len(loaded_splits[1])}")

    dataset_train = datasets.AISDataset(
        loaded_splits[0], max_seqlen=config.max_seqlen + 1, device=device)
    dataset_validation = datasets.AISDataset(
        loaded_splits[1], max_seqlen=config.max_seqlen + 1, device=device)

    dl_train = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True)
    dl_validation = DataLoader(
        dataset_validation, batch_size=config.batch_size, shuffle=False)

    return {"train": dl_train, "test": dl_validation}, dataset_train, dataset_validation, roi


def prepare_test_dataloaders(config: TestConfig, dataset_path: str, roi: ROI):
    device = torch.device(config.device)

    loaded_data_test, _ = load_data(
        [1.0], dataset_path, config.token_interval_seconds, config.min_seqlen, roi=roi
    )

    print(f"Test: {len(loaded_data_test[0])}")

    dataset_test = datasets.AISDataset(
        loaded_data_test[0], max_seqlen=config.max_seqlen + 1, device=device)
    dl_test = DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=False)

    return dl_test

# ==========================================
# 3. CORE EXECUTION LOGIC
# ==========================================


def execute_training(config: TrainConfig, dls: dict, ds_train, ds_val, roi: ROI, ckpt_path: str):
    device = torch.device(config.device)
    config.final_tokens = 2 * len(ds_train) * config.max_seqlen
    config.warmup_tokens = config.final_tokens // 10
    model = models.TrAISformer(config, roi, partition_model=None).to(device)

    savedir = os.path.dirname(ckpt_path) or "."
    os.makedirs(savedir, exist_ok=True)

    trainer = trainers.Trainer(
        model,
        ds_train,
        ds_val,
        config,
        device=device,
        aisdls=dls,
        INIT_SEQLEN=config.init_seqlen,
        ckpt_path=ckpt_path,
        savedir=savedir
    )
    trainer.train()


def execute_testing(config: TestConfig, dl_test, ckpt_path: str, predictions_out: str):
    device = torch.device(config.device)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError(
            f"Checkpoint at {ckpt_path} is a raw state_dict, not a full checkpoint. "
            "This was likely saved by the final-epoch save, not save_checkpoint(). Retrain."
        )

    train_config = TrainConfig(**checkpoint['config'])
    roi = ROI(**checkpoint['roi'])
    model = models.TrAISformer(train_config, roi, partition_model=None)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    v_roi_min_tensor = torch.tensor(
        [roi.lat_min, roi.lon_min, roi.sog_min, roi.cog_min]).to(device)
    v_ranges_tensor = torch.tensor([
        roi.lat_max - roi.lat_min, roi.lon_max - roi.lon_min,
        roi.sog_max - roi.sog_min, roi.cog_max - roi.cog_min
    ]).to(device)

    all_predictions = []
    with torch.no_grad():
        for seqs, masks, seqlens, mmsis, time_starts in tqdm(dl_test):
            seqs_init = seqs[:, :config.init_seqlen, :].to(device)
            preds = trainers.sample(
                model, seqs_init, config.max_seqlen - config.init_seqlen,
                temperature=1.0, sample=False, sample_mode=config.sample_mode,
                r_vicinity=config.r_vicinity, top_k=config.top_k
            )

            # Full sequence mask (historic + future)
            masks = masks[:, :config.max_seqlen].to(device)

            # Denormalize full sequence (historic + future)
            real_coords = (preds[:, :, :2] *
                           v_ranges_tensor[:2] + v_roi_min_tensor[:2])
            real_coords[masks == 0] = float('nan')

            # Timestamps for full sequence starting from trajectory start
            T_total = real_coords.shape[1]
            relative_times = torch.arange(
                T_total, device=device) * config.token_interval_seconds
            all_timestamps = time_starts.to(
                device).unsqueeze(1) + relative_times

            combined = torch.cat(
                [real_coords, all_timestamps.unsqueeze(-1)], dim=-1)
            all_predictions.append(combined.cpu())

    # [N, T, 3] (lat, lon, timestamp)
    final_preds = torch.cat(all_predictions, dim=0).numpy()

    predictions = DatasetPredictions(
        lats=final_preds[:, :, 0],
        lons=final_preds[:, :, 1],
        timestamps=final_preds[:, :, 2],
        predictor_name=f"TrAISformer_{train_config.name}",
        num_historic_tokens=config.init_seqlen
    )
    predictions.save(predictions_out)

# ==========================================
# 4. HIGH-LEVEL PIPELINES
# ==========================================


def train_pipeline(config, dataset_path: str):
    print(f"Training with {config}...")

    model_path = f"Trained/{config.name}.pt"
    savedir = os.path.dirname(model_path) or "."
    os.makedirs(savedir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),                              # print to console
            logging.FileHandler(os.path.join(
                savedir, "log.txt"))  # and to file
        ]
    )

    dls, ds_train, ds_val, roi = prepare_train_dataloaders(
        config, dataset_path)
    execute_training(config, dls, ds_train, ds_val, roi, model_path)


def test_pipeline(config, dataset_path: str, model_path: str):
    print(f"Testing with {config}...")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()]
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    roi = ROI(**checkpoint['roi'])

    output_path = os.path.join("Predictions", os.path.splitext(
        os.path.basename(model_path))[0])

    dl_test = prepare_test_dataloaders(config, dataset_path, roi=roi)
    execute_testing(config, dl_test, model_path, output_path)


# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_model_parser = subparsers.add_parser("train")
    train_model_parser.add_argument("-cfg", "--config", required=True)
    train_model_parser.add_argument("-dsp", "--dataset-path", required=True)

    test_model_parser = subparsers.add_parser("test")
    test_model_parser.add_argument("-cfg", "--config", required=True)
    test_model_parser.add_argument("-dsp", "--dataset-path", required=True)
    test_model_parser.add_argument("-mp", "--model-path", required=True)

    args = parser.parse_args()

    if args.mode == "train":
        cfg = parse_config(args.config, TrainConfig)
        train_pipeline(cfg, args.dataset_path)
    elif args.mode == "test":
        cfg = parse_config(args.config, TestConfig)
        test_pipeline(cfg, args.dataset_path, args.model_path)
