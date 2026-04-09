import argparse
import os
import contextlib
import numpy as np
import torch
from data_handler import load_data, ROI
from dataclasses import asdict, dataclass
import yaml
import datasets
import models
import trainers
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

# ==========================================
# 1. CONFIGURATION
# ==========================================


@dataclass
class TrainConfig:
    name: str = "trAISformer"
    device: str = "cpu"
    track_mlflow: bool = False
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


@dataclass
class TestConfig:
    track_mlflow: bool = False
    device: str = "cpu"
    batch_size: int = 32
    token_interval_seconds: int = 600
    init_seqlen: int = 18
    max_seqlen: int = 120
    min_seqlen: int = 36
    sample_mode: str = "pos_vicinity"
    r_vicinity: int = 40
    top_k: int = 1


def load_config(config_path: str, config_class):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return config_class(**data)

# ==========================================
# 2. DATA PREPARATION
# ==========================================


def prepare_train_dataloaders(config: TrainConfig, dataset_path: str):
    device = torch.device(config.device)
    loaded_splits, roi = load_data(
        [config.train_split, 1 -
            config.train_split], dataset_path, config.token_interval_seconds, config.min_seqlen
    )

    dataset_train = datasets.AISDataset(
        loaded_splits[0], max_seqlen=config.max_seqlen + 1, device=device)
    dataset_validation = datasets.AISDataset(
        loaded_splits[1], max_seqlen=config.max_seqlen + 1, device=device)

    dl_train = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True)
    dl_validation = DataLoader(
        dataset_validation, batch_size=config.batch_size, shuffle=False)

    return {"train": dl_train, "test": dl_validation}, dataset_train, dataset_validation, roi


def prepare_test_dataloaders(config: TestConfig, dataset_path: str):
    device = torch.device(config.device)
    loaded_data_test, _ = load_data(
        [1.0], dataset_path, config.token_interval_seconds, config.min_seqlen
    )

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
        mlflow_active=config.track_mlflow,
        ckpt_path=ckpt_path,
        savedir=savedir
    )
    trainer.train()


def execute_testing(config: TestConfig, dl_test, ckpt_path: str, predictions_out: str):
    device = torch.device(config.device)
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

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

            masks = masks[:, :config.max_seqlen].to(device)
            future_mask = masks[:, config.init_seqlen:].unsqueeze(-1)

            real_preds = (preds[:, config.init_seqlen:, :2]
                          * v_ranges_tensor[:2] + v_roi_min_tensor[:2])
            real_preds[future_mask == 0] = float('nan')

            batch_size, T_future, _ = real_preds.shape
            start_offset = config.init_seqlen * config.token_interval_seconds
            relative_times = torch.arange(
                T_future, device=device) * config.token_interval_seconds
            prediction_timestamps = time_starts.to(
                device).unsqueeze(1) + start_offset + relative_times

            combined = torch.cat(
                [real_preds, prediction_timestamps.unsqueeze(-1)], dim=-1)
            all_predictions.append(combined.cpu())

    final_preds = torch.cat(all_predictions, dim=0).numpy()
    os.makedirs(os.path.dirname(predictions_out) or ".", exist_ok=True)
    np.save(predictions_out, final_preds)
    print(f"Predictions saved locally to {predictions_out}")

# ==========================================
# 4. HIGH-LEVEL PIPELINES (With MLFlow Context)
# ==========================================


def train_pipeline(config_path: str, dataset_path: str, experiment_name: str, ckpt_path: str, artifact_id: str | None = None):
    config = load_config(config_path, TrainConfig)
    print(f"Training with {config}...")

    run_context = mlflow.start_run() if config.track_mlflow else contextlib.nullcontext()

    with run_context:
        if config.track_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.log_params(asdict(config))
            mlflow.log_param("dataset_used", dataset_path)
            if artifact_id is not None:
                mlflow.set_tag("artifact_id", artifact_id)
                mlflow.set_tag("artifact_type", "model")

        dls, ds_train, ds_val, roi = prepare_train_dataloaders(
            config, dataset_path)
        execute_training(config, dls, ds_train, ds_val, roi, ckpt_path)

        if config.track_mlflow:
            mlflow.log_artifact(ckpt_path)


def test_pipeline(config_path: str, dataset_path: str, experiment_name: str, ckpt_path: str, predictions_out: str, artifact_id: str | None = None):
    config = load_config(config_path, TestConfig)
    print(f"Testing with {config}...")

    run_context = mlflow.start_run() if config.track_mlflow else contextlib.nullcontext()

    with run_context:
        if config.track_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.log_params(asdict(config))
            mlflow.log_param("dataset_used", dataset_path)
            if artifact_id is not None:
                mlflow.set_tag("artifact_id", artifact_id)
                mlflow.set_tag("artifact_type", "predictions")

        dl_test = prepare_test_dataloaders(config, dataset_path)
        execute_testing(config, dl_test, ckpt_path, predictions_out)

        if config.track_mlflow:
            mlflow.log_artifact(predictions_out, artifact_path="predictions")


# ==========================================
# ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment_name", required=False,
                        default="Default_Experiment")
    parser.add_argument("--ckpt_path", required=False,
                        help="Override model save path", default="models/model.pt")
    parser.add_argument("--predictions_out", required=False,
                        help="Override predictions save path", default="predictions/predictions.npz")
    parser.add_argument("--artifact_id", required=False)

    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline(args.config, args.dataset,
                       args.experiment_name, args.ckpt_path, args.artifact_id)
    elif args.mode == "test":
        test_pipeline(args.config, args.dataset, args.experiment_name,
                      args.ckpt_path, args.predictions_out, args.artifact_id)
