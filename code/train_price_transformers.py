from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_price_pytorch import (
    FEATURE_COLS,
    MACRO_FEATURE_COLS,
    TARGET_COL,
    build_dataloaders,
    evaluate_test,
    forecast_future,
    load_data,
    run_epoch,
    run_shap_kernel_explainer,
    split_dataset_by_date,
)


@dataclass
class TransformerTrainConfig:
    bank_name: str
    forecast_days: int = 30
    data_dir: Path = Path("data")
    output_root: Path = Path("results_transformer")
    time_steps: int = 60
    epochs: int = 50
    batch_size: int = 32
    hidden_size: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    model_type: str = "informer"
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: int = 4
    moving_avg: int = 5
    distill: bool = True
    optimize: bool = False
    use_macro: bool = False
    tune_epochs: int = 8
    tune_candidates: int = 6
    hpo_method: str = "ga"
    hpo_generations: int = 5
    run_shap: bool = False
    shap_only: bool = False
    save_model: bool = False
    model_path: Path | None = None
    shap_background_size: int = 64
    shap_eval_size: int = 32
    shap_nsamples: int = 100
    seed: int = 42


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProbSparseSelfAttention(nn.Module):
    """Informer ProbSparse self-attention adapted for batched encoder inputs."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float, factor: int = 5) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.factor = factor
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        sample_k = min(seq_len, max(1, int(self.factor * math.log(seq_len + 1))))
        n_top = min(seq_len, max(1, int(self.factor * math.log(seq_len + 1))))

        if sample_k < seq_len:
            index_sample = torch.randint(seq_len, (sample_k,), device=x.device)
            k_sample = k[:, :, index_sample, :]
        else:
            k_sample = k

        sample_scores = torch.matmul(q, k_sample.transpose(-2, -1)) * self.scale
        sparsity = sample_scores.max(dim=-1).values - sample_scores.mean(dim=-1)
        top_index = sparsity.topk(n_top, dim=-1).indices

        context = v.mean(dim=-2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        gather_index = top_index.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        q_top = torch.gather(q, dim=2, index=gather_index)
        scores_top = torch.matmul(q_top, k.transpose(-2, -1)) * self.scale
        attn_top = self.dropout(torch.softmax(scores_top, dim=-1))
        context_top = torch.matmul(attn_top, v)
        context.scatter_(2, gather_index, context_top)

        out = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)


class InformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.attention = ProbSparseSelfAttention(
            config.hidden_size,
            config.n_heads,
            config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * config.ff_multiplier, config.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class Informer(nn.Module):
    """Informer baseline with ProbSparse attention and encoder distilling."""

    def __init__(self, n_features: int, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.hidden_size)
        self.position = PositionalEncoding(config.hidden_size)
        self.layers = nn.ModuleList([InformerEncoderLayer(config) for _ in range(config.n_layers)])
        self.distill = config.distill
        self.distillers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for _ in range(max(0, config.n_layers - 1))
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.position(self.input_projection(x))
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if self.distill and idx < len(self.distillers) and h.size(1) > 2:
                h = self.distillers[idx](h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1]).squeeze(-1)


class InformerLite(nn.Module):
    """Practical Informer-style baseline using attention plus sequence distilling."""

    def __init__(self, n_features: int, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_features, config.hidden_size)
        self.position = PositionalEncoding(config.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.hidden_size * config.ff_multiplier,
                    dropout=config.dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(config.n_layers)
            ]
        )
        self.distill = config.distill
        self.distillers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                for _ in range(max(0, config.n_layers - 1))
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.position(self.input_projection(x))
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if self.distill and idx < len(self.distillers) and h.size(1) > 2:
                h = self.distillers[idx](h.transpose(1, 2)).transpose(1, 2)
        return self.head(h[:, -1]).squeeze(-1)


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = max(1, int(kernel_size))
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kernel_size == 1:
            trend = x
        else:
            left = (self.kernel_size - 1) // 2
            right = self.kernel_size - 1 - left
            padded = torch.nn.functional.pad(x.transpose(1, 2), (left, right), mode="replicate")
            trend = self.pool(padded).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """Autoformer Auto-Correlation mechanism with FFT-based period discovery."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float, factor: int = 3) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.factor = factor
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q_fft = torch.fft.rfft(q, dim=2)
        k_fft = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=seq_len, dim=2)
        corr_score = corr.mean(dim=(1, 3))
        top_k = min(seq_len, max(1, int(self.factor * math.log(seq_len + 1))))
        delays = corr_score.mean(dim=0).topk(top_k).indices
        weights = torch.softmax(corr_score[:, delays], dim=-1)
        weights = self.dropout(weights)

        aggregated = torch.zeros_like(v)
        for i, delay in enumerate(delays):
            shifted = torch.roll(v, shifts=-int(delay.item()), dims=2)
            weight = weights[:, i].view(bsz, 1, 1, 1)
            aggregated = aggregated + shifted * weight

        out = aggregated.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, n_features: int, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.decompose1 = MovingAverageDecomposition(config.moving_avg)
        self.decompose2 = MovingAverageDecomposition(config.moving_avg)
        self.autocorr = AutoCorrelation(config.hidden_size, config.n_heads, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * config.ff_multiplier, config.hidden_size),
        )
        self.trend_projection = nn.Linear(config.hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x + self.dropout(self.autocorr(x))
        seasonal, trend1 = self.decompose1(h)
        h = seasonal + self.dropout(self.ffn(seasonal))
        seasonal, trend2 = self.decompose2(h)
        trend = self.trend_projection(trend1 + trend2)
        return seasonal, trend


class Autoformer(nn.Module):
    """Autoformer baseline with decomposition blocks and Auto-Correlation."""

    def __init__(self, n_features: int, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.initial_decompose = MovingAverageDecomposition(config.moving_avg)
        self.seasonal_projection = nn.Linear(n_features, config.hidden_size)
        self.position = PositionalEncoding(config.hidden_size)
        self.layers = nn.ModuleList([AutoformerEncoderLayer(n_features, config) for _ in range(config.n_layers)])
        self.seasonal_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(n_features, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.initial_decompose(x)
        h = self.position(self.seasonal_projection(seasonal))
        trend_terms = [trend]
        for layer in self.layers:
            h, trend_update = layer(h)
            trend_terms.append(trend_update)
        trend_total = torch.stack(trend_terms, dim=0).sum(dim=0)
        seasonal_out = self.seasonal_head(h[:, -1]).squeeze(-1)
        trend_out = self.trend_head(trend_total.mean(dim=1)).squeeze(-1)
        return seasonal_out + trend_out


class AutoformerLite(nn.Module):
    """Autoformer-style baseline with moving-average decomposition and seasonal mixing."""

    def __init__(self, n_features: int, config: TransformerTrainConfig) -> None:
        super().__init__()
        self.decompose = MovingAverageDecomposition(config.moving_avg)
        self.seasonal_projection = nn.Linear(n_features, config.hidden_size)
        self.position = PositionalEncoding(config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seasonal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.seasonal_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(n_features, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decompose(x)
        seasonal_h = self.position(self.seasonal_projection(seasonal))
        seasonal_h = self.seasonal_encoder(seasonal_h)
        seasonal_out = self.seasonal_head(seasonal_h[:, -1]).squeeze(-1)
        trend_out = self.trend_head(trend.mean(dim=1)).squeeze(-1)
        return seasonal_out + trend_out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(n_features: int, config: TransformerTrainConfig) -> nn.Module:
    if config.model_type == "informer":
        return Informer(n_features, config)
    if config.model_type == "autoformer":
        return Autoformer(n_features, config)
    if config.model_type == "informer_lite":
        return InformerLite(n_features, config)
    if config.model_type == "autoformer_lite":
        return AutoformerLite(n_features, config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: TransformerTrainConfig,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        train_seconds = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_seconds = time.perf_counter() - val_start
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"lr={lr:.6g} | train_s={train_seconds:.2f} | val_s={val_seconds:.2f} | "
            f"epoch_s={epoch_seconds:.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_candidate(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: list[str],
    trial: TransformerTrainConfig,
    device: torch.device,
) -> float:
    train_loader, val_loader, *_ = build_dataloaders(
        train_set, val_set, val_set, feature_cols, trial.time_steps, trial.batch_size
    )
    model = build_model(len(feature_cols), trial).to(device)
    _, history = train_model(model, train_loader, val_loader, trial, device)
    return min(h["val_loss"] for h in history)


def tune_hyperparams_random(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: list[str],
    base_config: TransformerTrainConfig,
    device: torch.device,
) -> TransformerTrainConfig:
    print("Running lightweight random search for transformer baseline...")
    search_space = {
        "hidden_size": [16, 32, 64],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [16, 32, 64],
        "dropout": [0.2, 0.3, 0.4],
    }
    best_val = float("inf")
    best_cfg = copy.deepcopy(base_config)

    for i in range(1, max(1, base_config.tune_candidates) + 1):
        trial = copy.deepcopy(base_config)
        trial.hidden_size = random.choice(search_space["hidden_size"])
        trial.learning_rate = random.choice(search_space["learning_rate"])
        trial.batch_size = random.choice(search_space["batch_size"])
        trial.dropout = random.choice(search_space["dropout"])
        trial.epochs = max(1, int(base_config.tune_epochs))
        trial_best_val = evaluate_candidate(train_set, val_set, feature_cols, trial, device)
        print(
            f"Trial {i}/{base_config.tune_candidates} | val={trial_best_val:.6f} | "
            f"hidden={trial.hidden_size}, layers={trial.n_layers}, lr={trial.learning_rate}, "
            f"bs={trial.batch_size}, drop={trial.dropout}"
        )
        if trial_best_val < best_val:
            best_val = trial_best_val
            best_cfg = copy.deepcopy(trial)

    best_cfg.epochs = base_config.epochs
    print(f"Best transformer random-search val_loss: {best_val:.6f}")
    return best_cfg


def tune_hyperparams_ga(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: list[str],
    base_config: TransformerTrainConfig,
    device: torch.device,
) -> TransformerTrainConfig:
    print("Running genetic-algorithm HPO for transformer baseline...")
    search_space = {
        "hidden_size": [32, 64, 96, 128],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "batch_size": [16, 32, 64],
        "dropout": [0.1, 0.2, 0.3],
        "time_steps": [10, 20, 30, 60],
        "n_layers": [1, 2, 3],
    }
    population_size = max(2, int(base_config.tune_candidates))
    generations = max(1, int(base_config.hpo_generations))
    gene_names = list(search_space.keys())

    def sample_gene(name: str) -> int | float:
        values = search_space[name]
        if name == "hidden_size":
            values = [v for v in values if int(v) % base_config.n_heads == 0]
        return random.choice(values)

    def make_trial(genes: dict[str, int | float]) -> TransformerTrainConfig:
        trial = copy.deepcopy(base_config)
        trial.hidden_size = int(genes["hidden_size"])
        trial.learning_rate = float(genes["learning_rate"])
        trial.batch_size = int(genes["batch_size"])
        trial.dropout = float(genes["dropout"])
        trial.time_steps = int(genes["time_steps"])
        trial.n_layers = int(genes["n_layers"])
        trial.epochs = max(1, int(base_config.tune_epochs))
        return trial

    def fitness(genes: dict[str, int | float]) -> tuple[float, TransformerTrainConfig]:
        trial = make_trial(genes)
        val_loss = evaluate_candidate(train_set, val_set, feature_cols, trial, device)
        return val_loss, trial

    def crossover(a: dict[str, int | float], b: dict[str, int | float]) -> dict[str, int | float]:
        cut = random.randint(1, len(gene_names) - 1)
        return {
            name: a[name] if idx < cut else b[name]
            for idx, name in enumerate(gene_names)
        }

    def mutate(genes: dict[str, int | float], probability: float = 0.2) -> dict[str, int | float]:
        out = dict(genes)
        for name in gene_names:
            if random.random() < probability:
                out[name] = sample_gene(name)
        return out

    population = [
        {name: sample_gene(name) for name in gene_names}
        for _ in range(population_size)
    ]
    best_val = float("inf")
    best_cfg = copy.deepcopy(base_config)

    for generation in range(1, generations + 1):
        scored: list[tuple[float, dict[str, int | float], TransformerTrainConfig]] = []
        for idx, genes in enumerate(population, start=1):
            val_loss, trial = fitness(genes)
            scored.append((val_loss, genes, trial))
            print(
                f"GA gen={generation}/{generations} candidate={idx}/{population_size} | "
                f"val={val_loss:.6f} | hidden={trial.hidden_size}, layers={trial.n_layers}, "
                f"steps={trial.time_steps}, lr={trial.learning_rate}, bs={trial.batch_size}, "
                f"drop={trial.dropout}"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_cfg = copy.deepcopy(trial)

        scored.sort(key=lambda item: item[0])
        elites = [dict(item[1]) for item in scored[: max(1, population_size // 4)]]
        next_population = elites.copy()
        while len(next_population) < population_size:
            tournament_a = min(random.sample(scored, k=min(3, len(scored))), key=lambda item: item[0])[1]
            tournament_b = min(random.sample(scored, k=min(3, len(scored))), key=lambda item: item[0])[1]
            next_population.append(mutate(crossover(tournament_a, tournament_b)))
        population = next_population

    best_cfg.epochs = base_config.epochs
    print(f"Best transformer GA val_loss: {best_val:.6f}")
    return best_cfg


def tune_hyperparams(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feature_cols: list[str],
    base_config: TransformerTrainConfig,
    device: torch.device,
) -> TransformerTrainConfig:
    if base_config.hpo_method == "ga":
        return tune_hyperparams_ga(train_set, val_set, feature_cols, base_config, device)
    if base_config.hpo_method == "random":
        return tune_hyperparams_random(train_set, val_set, feature_cols, base_config, device)
    raise ValueError(f"Unsupported hpo_method: {base_config.hpo_method}")


def parse_args() -> TransformerTrainConfig:
    parser = argparse.ArgumentParser(description="Train Informer/Autoformer-style stock-price baselines.")
    parser.add_argument("--bank", type=str, help="Bank symbol, e.g. BBCA")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-root", type=str, default="results_transformer")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--use-macro", action="store_true")
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-candidates", type=int, default=6)
    parser.add_argument("--hpo-method", choices=["ga", "random"], default="ga")
    parser.add_argument("--hpo-generations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--model-type",
        choices=["informer", "autoformer", "informer_lite", "autoformer_lite"],
        default="informer",
    )
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--moving-avg", type=int, default=5)
    parser.add_argument("--no-distill", action="store_true")
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--shap-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--shap-background-size", type=int, default=64)
    parser.add_argument("--shap-eval-size", type=int, default=32)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bank_name = args.bank.strip().upper() if args.bank else input("Enter bank name (e.g., BBCA): ").strip().upper()
    return TransformerTrainConfig(
        bank_name=bank_name,
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        forecast_days=args.forecast_days,
        epochs=args.epochs,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_multiplier=args.ff_multiplier,
        moving_avg=args.moving_avg,
        distill=not args.no_distill,
        optimize=args.optimize,
        use_macro=args.use_macro,
        tune_epochs=args.tune_epochs,
        tune_candidates=args.tune_candidates,
        hpo_method=args.hpo_method,
        hpo_generations=args.hpo_generations,
        run_shap=args.run_shap,
        shap_only=args.shap_only,
        save_model=args.save_model,
        model_path=Path(args.model_path) if args.model_path else None,
        shap_background_size=args.shap_background_size,
        shap_eval_size=args.shap_eval_size,
        shap_nsamples=args.shap_nsamples,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    if config.hidden_size % config.n_heads != 0:
        raise ValueError("--hidden-size must be divisible by --n-heads.")
    if config.shap_only and not config.run_shap:
        raise ValueError("`--shap-only` requires `--run-shap`.")
    if config.shap_only and config.optimize:
        raise ValueError("`--shap-only` cannot be combined with `--optimize`.")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = FEATURE_COLS + MACRO_FEATURE_COLS if config.use_macro else FEATURE_COLS
    print(f"Device: {device}")
    print(f"Bank: {config.bank_name} | Model type: {config.model_type}")
    print(f"Data dir: {config.data_dir} | Use macro: {config.use_macro}")
    print(f"Feature count: {len(feature_cols)}")

    output_dir = config.output_root / config.bank_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_data, _, target_scaler = load_data(config, feature_cols)
    train_set = split_dataset_by_date(selected_data, "2014-01-02", "2020-01-02")
    val_set = split_dataset_by_date(selected_data, "2020-01-02", "2022-01-02")
    test_set = split_dataset_by_date(selected_data, "2022-01-02", "2024-12-31")
    print(f"Split sizes | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    if config.optimize and not config.shap_only:
        config = tune_hyperparams(train_set, val_set, feature_cols, config, device)

    train_loader, val_loader, x_train, _, x_test, y_test = build_dataloaders(
        train_set, val_set, test_set, feature_cols, config.time_steps, config.batch_size
    )
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Insufficient sequence samples. Reduce --time-steps or verify date ranges in data.")

    model = build_model(len(feature_cols), config).to(device)
    history: list[dict[str, float]] = []
    model_file = config.model_path if config.model_path is not None else output_dir / "model_transformer.pt"

    if config.shap_only:
        if not model_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
        payload = torch.load(model_file, map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model checkpoint: {model_file}")
    else:
        model, history = train_model(model, train_loader, val_loader, config, device)
        if config.save_model:
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "model_type": config.model_type,
                "hidden_size": config.hidden_size,
                "dropout": config.dropout,
                "time_steps": config.time_steps,
                "use_macro": config.use_macro,
                "feature_columns": feature_cols,
                "bank_name": config.bank_name,
                "seed": config.seed,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "ff_multiplier": config.ff_multiplier,
                "moving_avg": config.moving_avg,
                "distill": config.distill,
            }
            torch.save(
                checkpoint_payload,
                model_file,
            )
            torch.save(checkpoint_payload, output_dir / "model.pt")
            print(f"Saved model checkpoint: {model_file}")

    metrics, y_true, y_pred = evaluate_test(model, x_test, y_test, target_scaler, device)
    print(
        f"Test metrics | MSE={metrics['mse']:.4f} RMSE={metrics['rmse']:.4f} "
        f"MAE={metrics['mae']:.4f} MAPE={metrics['mape']:.2f}%"
    )

    forecasts = forecast_future(model, x_test[-1], config.forecast_days, target_scaler, device)
    test_dates = test_set.index[config.time_steps:]
    future_dates = pd.date_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=config.forecast_days, freq="D")

    pd.DataFrame(
        {"date": test_dates, "actual": y_true, "predicted": y_pred, "abs_error": np.abs(y_true - y_pred)}
    ).to_csv(output_dir / "test_predictions_transformer.csv", index=False)
    pd.DataFrame(
        {"date": test_dates, "actual": y_true, "predicted": y_pred, "abs_error": np.abs(y_true - y_pred)}
    ).to_csv(output_dir / "test_predictions_pytorch.csv", index=False)
    pd.DataFrame({"date": future_dates, "forecast_price": forecasts}).to_csv(
        output_dir / "forecast_transformer.csv", index=False
    )
    pd.DataFrame({"date": future_dates, "forecast_price": forecasts}).to_csv(
        output_dir / "forecast_pytorch.csv", index=False
    )
    if history:
        pd.DataFrame(history).to_csv(output_dir / "training_history_transformer.csv", index=False)
        pd.DataFrame(history).to_csv(output_dir / "training_history_pytorch.csv", index=False)

    shap_importance = None
    if config.run_shap:
        shap_importance = run_shap_kernel_explainer(
            model=model,
            x_train=x_train,
            x_test=x_test,
            feature_cols=feature_cols,
            time_steps=config.time_steps,
            device=device,
            background_size=config.shap_background_size,
            eval_size=config.shap_eval_size,
            nsamples=config.shap_nsamples,
        )
        shap_importance.to_csv(output_dir / "shap_kernel_feature_importance_transformer.csv", index=False)
        shap_importance.to_csv(output_dir / "shap_kernel_feature_importance.csv", index=False)

    config_dict = asdict(config)
    config_dict["data_dir"] = str(config_dict["data_dir"])
    config_dict["output_root"] = str(config_dict["output_root"])
    config_dict["model_path"] = str(config_dict["model_path"]) if config_dict["model_path"] is not None else None
    summary = {
        "config": config_dict,
        "metrics": metrics,
        "n_train_sequences": int(len(x_train)),
        "n_test_sequences": int(len(x_test)),
        "feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "architecture_note": (
            "informer uses ProbSparse self-attention with encoder distilling; "
            "autoformer uses moving-average decomposition blocks with FFT-based Auto-Correlation; "
            "informer_lite and autoformer_lite keep the earlier lightweight style baselines."
        ),
    }
    if shap_importance is not None:
        summary["shap"] = {
            "method": "kernel_explainer",
            "background_size": int(min(config.shap_background_size, len(x_train))),
            "eval_size": int(min(config.shap_eval_size, len(x_test))),
            "nsamples": int(config.shap_nsamples),
            "top_feature": str(shap_importance.iloc[0]["feature"]),
        }
    with open(output_dir / "summary_transformer.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    with open(output_dir / "summary_pytorch.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(" - test_predictions_transformer.csv")
    print(" - test_predictions_pytorch.csv")
    print(" - forecast_transformer.csv")
    print(" - forecast_pytorch.csv")
    if history:
        print(" - training_history_transformer.csv")
        print(" - training_history_pytorch.csv")
    print(" - summary_transformer.json")
    print(" - summary_pytorch.json")
    if config.save_model and not config.shap_only:
        print(" - model_transformer.pt")
        print(" - model.pt")
    if shap_importance is not None:
        print(" - shap_kernel_feature_importance_transformer.csv")
        print(" - shap_kernel_feature_importance.csv")


if __name__ == "__main__":
    main()
