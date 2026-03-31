import argparse
import torch
from data_handler import load_data
from dataclasses import dataclass
import yaml
import datasets
import models
import trainers
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class Config:
    device: str = "cpu"
    path_data_train: str = "data/train.npz"
    path_data_validation: str = "data/validation.npz"
    path_data_test: str = "data/test.npz"
    ckpt_path: str = "models/model.pt"

    num_workers: int = 0

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

    sample_mode: str = "pos_vicinity"
    r_vicinity: int = 40
    top_k: int = 1


def train(config: Config):
    print(f"Training with {config}...")
    device = torch.device(config.device)

    data_train, _ = load_data(config.path_data_train,
                              config.token_interval_seconds, config.min_seqlen)
    data_validation, _ = load_data(
        config.path_data_validation, config.token_interval_seconds, config.min_seqlen)

    dataset_train = datasets.AISDataset(
        data_train, max_seqlen=config.max_seqlen + 1, device=device)
    dataset_valdiation = datasets.AISDataset(
        data_validation, max_seqlen=config.max_seqlen + 1, device=device)

    dl_train = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=False)
    dl_validation = DataLoader(
        dataset_valdiation, batch_size=config.batch_size, shuffle=False)

    model = models.TrAISformer(config, partition_model=None)
    model.to(config.device)

    trainer = trainers.Trainer(
        model,
        dataset_train,
        dataset_valdiation,
        config,
        device=device,
        aisdls={"train": dl_train, "validation": dl_validation},
        INIT_SEQLEN=config.init_seqlen
    )

    trainer.train()


def test(config: Config):
    print(f"Testing with {config}...")
    device = torch.device(config.device)

    _, roi = load_data(config.path_data_train,
                       config.token_interval_seconds, config.min_seqlen)
    data_test, _ = load_data(config.path_data_test,
                             config.token_interval_seconds, config.min_seqlen)
    dataset_test = datasets.AISDataset(
        data_test, max_seqlen=config.max_seqlen + 1, device=device)
    dl_test = DataLoader(
        dataset_test, batch_size=config.batch_size, shuffle=False)

    model = models.TrAISformer(config, partition_model=None)
    model.load_state_dict(torch.load(
        config.ckpt_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    v_roi_min_tensor = torch.tensor(
        [roi.lat_min, roi.lon_min, roi.sog_min, roi.cog_min]).to(config.device)
    v_ranges_tensor = torch.tensor([roi.lat_max - roi.lat_min, roi.lon_max - roi.lon_min,
                                   roi.sog_max - roi.sog_min, roi.cog_max - roi.cog_min]).to(config.device)

    all_predictions = []

    pbar = tqdm(enumerate(dl_test), total=len(dl_test))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :config.init_seqlen, :].to(config.device)

            preds = trainers.sample(
                model,
                seqs_init,
                config.max_seqlen - config.init_seqlen,
                temperature=1.0,
                sample=False,
                sample_mode=config.sample_mode,
                r_vicinity=config.r_vicinity,
                top_k=config.top_k
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

    return final_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = yaml.safe_load(f)
        config = Config(**data)

    if args.mode == "train":
        train(config)

    elif args.mode == "test":
        test(config)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")
