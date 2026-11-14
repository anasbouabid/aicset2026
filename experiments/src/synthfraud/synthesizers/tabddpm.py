"""TabDDPM backbone (Kotelnikov et al., ICML 2023).

Gaussian diffusion on quantile-normalized numerics plus multinomial diffusion on
categoricals, including the label as a categorical column (joint table, same as
CTGAN/TVAE — not class-conditional sampling).

synthcity is not used: it pins ``torch<2.3`` and ``numpy<2``, which would smash
the working CTGAN stack. Official yandex-research/tab-ddpm pins numpy 1.21.
This module uses the torch already in ``synthfraud``. On macOS, set
``KMP_DUPLICATE_LIB_OK=TRUE`` before importing torch (duplicate libomp from
conda-forge OpenBLAS + pip torch). That is the documented OpenMP workaround,
not a second env.
"""

from __future__ import annotations

import math
import os

# Must run before torch is imported in this process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

from synthfraud.log import log
from synthfraud.schema import Schema


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cosine_betas(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2

    betas = []
    for i in range(timesteps):
        t1, t2 = i / timesteps, (i + 1) / timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def _log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _index_to_log_onehot(x: torch.Tensor, num_classes: np.ndarray) -> torch.Tensor:
    onehots = [F.one_hot(x[:, i], int(num_classes[i])) for i in range(len(num_classes))]
    x_onehot = torch.cat(onehots, dim=1)
    return torch.log(x_onehot.float().clamp(min=1e-30))


def _ohe_to_categories(ohe: torch.Tensor, num_classes: np.ndarray) -> torch.Tensor:
    offsets = np.concatenate([[0], np.cumsum(num_classes)])
    cols = [ohe[:, offsets[i] : offsets[i + 1]].argmax(dim=1) for i in range(len(num_classes))]
    return torch.stack(cols, dim=1)


class _MLPDiffusion(nn.Module):
    def __init__(self, d_in: int, hidden: list[int], dim_t: int = 128) -> None:
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        layers: list[nn.Module] = []
        d = dim_t
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        layers.append(nn.Linear(d, d_in))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y=None) -> torch.Tensor:
        emb = self.time_embed(_timestep_embedding(timesteps, self.dim_t))
        return self.mlp(self.proj(x) + emb)


class _GaussianMultinomialDiffusion(nn.Module):
    """Joint Gaussian (numeric) + multinomial (categorical) reverse process."""

    def __init__(
        self,
        num_classes: np.ndarray,
        n_num: int,
        denoise_fn: nn.Module,
        num_timesteps: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_classes = np.asarray(num_classes, dtype=np.int64)
        self.n_num = int(n_num)
        self.num_timesteps = int(num_timesteps)
        self.device = device
        self._denoise_fn = denoise_fn

        offsets = np.concatenate([[0], np.cumsum(self.num_classes)])
        self.offsets = torch.tensor(offsets, device=device, dtype=torch.long)
        self.slices = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(self.num_classes))]
        expanded = np.concatenate(
            [np.full(int(k), int(k), dtype=np.float32) for k in self.num_classes]
        )
        self.num_classes_expanded = torch.tensor(expanded, device=device)

        betas = _cosine_betas(self.num_timesteps)
        alphas = 1.0 - betas
        log_alpha = torch.log(alphas.clamp(min=1e-12))
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)
        log_1_min_alpha = torch.log((1.0 - alphas).clamp(min=1e-12))
        log_1_min_cumprod_alpha = torch.log((1.0 - torch.exp(log_cumprod_alpha)).clamp(min=1e-12))
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-12)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]])
        )
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(
            min=1e-12
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod).clamp(min=1e-12)
        )

        def buf(name: str, t: torch.Tensor) -> None:
            self.register_buffer(name, t.float().to(device))

        buf("alphas", alphas)
        buf("log_alpha", log_alpha)
        buf("log_1_min_alpha", log_1_min_alpha)
        buf("log_cumprod_alpha", log_cumprod_alpha)
        buf("log_1_min_cumprod_alpha", log_1_min_cumprod_alpha)
        buf("alphas_cumprod", alphas_cumprod)
        buf("alphas_cumprod_prev", alphas_cumprod_prev)
        buf("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        buf("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        buf("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        buf("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        buf("posterior_variance", posterior_variance)
        buf("posterior_log_variance_clipped", posterior_log_variance_clipped)
        buf("posterior_mean_coef1", posterior_mean_coef1)
        buf("posterior_mean_coef2", posterior_mean_coef2)

    def _q_num(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def _q_cat(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_probs = _log_add_exp(
            log_x0 + _extract(self.log_cumprod_alpha, t, log_x0.shape),
            _extract(self.log_1_min_cumprod_alpha, t, log_x0.shape)
            - torch.log(self.num_classes_expanded),
        )
        return self._gumbel_sample(log_probs)

    def _gumbel_sample(self, logits: torch.Tensor) -> torch.Tensor:
        parts = []
        for sl in self.slices:
            block = logits[:, sl]
            u = torch.rand_like(block)
            gumbel = -torch.log(-torch.log(u + 1e-30) + 1e-30)
            parts.append((gumbel + block).argmax(dim=1, keepdim=True))
        idx = torch.cat(parts, dim=1)
        return _index_to_log_onehot(idx, self.num_classes)

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _q_posterior_mean_var(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x0
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def _sliced_lse(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for sl in self.slices:
            out[:, sl] = torch.logsumexp(x[:, sl], dim=1, keepdim=True)
        return out

    def _q_pred_one_step(self, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _log_add_exp(
            log_x_t + _extract(self.log_alpha, t, log_x_t.shape),
            _extract(self.log_1_min_alpha, t, log_x_t.shape) - torch.log(self.num_classes_expanded),
        )

    def _q_pred(self, log_x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _log_add_exp(
            log_x0 + _extract(self.log_cumprod_alpha, t, log_x0.shape),
            _extract(self.log_1_min_cumprod_alpha, t, log_x0.shape)
            - torch.log(self.num_classes_expanded),
        )

    def _q_posterior_cat(self, log_x0: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_prev = torch.where(t - 1 < 0, torch.zeros_like(t), t - 1)
        log_unnorm = self._q_pred(log_x0, t_prev) + self._q_pred_one_step(log_x_t, t)
        return log_unnorm - self._sliced_lse(log_unnorm)

    def _cat_x0(self, model_out_cat: torch.Tensor) -> torch.Tensor:
        log_pred = torch.empty_like(model_out_cat)
        for sl in self.slices:
            log_pred[:, sl] = F.log_softmax(model_out_cat[:, sl], dim=1)
        return log_pred

    def mixed_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device)
        x_num = x[:, : self.n_num]
        x_cat = x[:, self.n_num :]
        noise = torch.randn_like(x_num)
        x_num_t = self._q_num(x_num, t, noise)
        log_x0 = _index_to_log_onehot(x_cat.long(), self.num_classes)
        log_x_t = self._q_cat(log_x0, t)
        model_out = self._denoise_fn(torch.cat([x_num_t, log_x_t], dim=1), t)
        out_num = model_out[:, : self.n_num]
        out_cat = model_out[:, self.n_num :]
        loss_g = (noise - out_num).pow(2).mean()
        log_true = self._q_posterior_cat(log_x0, log_x_t, t)
        log_model = self._q_posterior_cat(self._cat_x0(out_cat), log_x_t, t)
        kl = (log_true.exp() * (log_true - log_model)).sum(dim=1)
        nll = -(log_x0.exp() * log_model).sum(dim=1)
        mask = (t == 0).float()
        loss_m = (mask * nll + (1.0 - mask) * kl).mean() / max(len(self.num_classes), 1)
        return loss_m, loss_g

    @torch.no_grad()
    def sample(self, n: int, batch_size: int = 2048) -> torch.Tensor:
        self.eval()
        chunks = []
        remaining = n
        while remaining > 0:
            b = min(batch_size, remaining)
            chunks.append(self._sample_batch(b))
            remaining -= b
        return torch.cat(chunks, dim=0)[:n]

    @torch.no_grad()
    def _sample_batch(self, b: int) -> torch.Tensor:
        z_num = torch.randn((b, self.n_num), device=self.device)
        uniform = torch.zeros((b, int(self.num_classes.sum())), device=self.device)
        log_z = self._gumbel_sample(uniform)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            out = self._denoise_fn(torch.cat([z_num, log_z], dim=1).float(), t)
            out_num, out_cat = out[:, : self.n_num], out[:, self.n_num :]
            x0 = self._predict_x0_from_eps(z_num, t, out_num)
            mean, _, log_var = self._q_posterior_mean_var(x0, z_num, t)
            noise = torch.randn_like(z_num)
            mask = (t != 0).float().view(-1, 1)
            z_num = mean + mask * torch.exp(0.5 * log_var) * noise
            log_model = self._q_posterior_cat(self._cat_x0(out_cat), log_z, t)
            log_z = self._gumbel_sample(log_model)
        z_cat = _ohe_to_categories(torch.exp(log_z).round(), self.num_classes)
        return torch.cat([z_num, z_cat.float()], dim=1).cpu()


class TabDDPMSynthesizer:
    def __init__(
        self,
        seed: int,
        schema: Schema,
        n_iter: int = 10000,
        num_timesteps: int = 1000,
        batch_size: int = 1024,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        hidden: list[int] | None = None,
        dim_t: int = 128,
        sample_batch: int = 2048,
    ) -> None:
        self.seed = int(seed)
        self.schema = schema
        self.n_iter = int(n_iter)
        self.num_timesteps = int(num_timesteps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden = list(hidden) if hidden else [256, 256, 256, 256]
        self.dim_t = int(dim_t)
        self.sample_batch = int(sample_batch)
        self.device = _device()
        self._diffusion: _GaussianMultinomialDiffusion | None = None
        self._qt: QuantileTransformer | None = None
        self._cat_levels: dict[str, list[str]] = {}
        self._round_cols: list[int] = []
        self._num_mins: np.ndarray | None = None
        self._num_maxs: np.ndarray | None = None

    def _encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_num = frame[list(self.schema.numeric)].to_numpy(dtype=np.float64)
        cat_cols = list(self.schema.categorical) + [self.schema.label]
        codes = []
        for col in cat_cols:
            levels = self._cat_levels[col]
            mapping = {v: i for i, v in enumerate(levels)}
            codes.append(frame[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64))
        x_cat = np.stack(codes, axis=1)
        return x_num, x_cat

    def fit(self, train: pd.DataFrame) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cols = self.schema.all_cols
        table = train[cols].copy()
        table[self.schema.label] = table[self.schema.label].astype(int).astype(str)
        for col in self.schema.categorical:
            table[col] = table[col].astype(str)
        cat_cols = list(self.schema.categorical) + [self.schema.label]
        self._cat_levels = {col: sorted(table[col].astype(str).unique().tolist()) for col in cat_cols}
        x_num, x_cat = self._encode(table)
        self._num_mins = x_num.min(axis=0)
        self._num_maxs = x_num.max(axis=0)
        self._round_cols = []
        for j in range(x_num.shape[1]):
            uniq = np.unique(x_num[:, j])
            if len(uniq) <= 32 and np.allclose(uniq, np.round(uniq)):
                self._round_cols.append(j)
        n_quantiles = max(min(len(table), 1000), 10)
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=int(1e9),
            random_state=self.seed,
        )
        x_num_t = self._qt.fit_transform(x_num).astype(np.float32)
        k = np.array([len(self._cat_levels[c]) for c in cat_cols], dtype=np.int64)
        d_in = x_num_t.shape[1] + int(k.sum())
        denoise = _MLPDiffusion(d_in, self.hidden, dim_t=self.dim_t).to(self.device)
        try:
            diffusion = _GaussianMultinomialDiffusion(
                k, x_num_t.shape[1], denoise, self.num_timesteps, self.device
            )
            diffusion.to(self.device)
            # Probe one MPS/CUDA op before committing to a long fit.
            _ = diffusion.mixed_loss(
                torch.zeros((2, x_num_t.shape[1] + x_cat.shape[1]), device=self.device)
            )
        except Exception as exc:
            log(f"TabDDPM device {self.device} failed ({exc!r}); falling back to CPU")
            self.device = torch.device("cpu")
            denoise = _MLPDiffusion(d_in, self.hidden, dim_t=self.dim_t).to(self.device)
            diffusion = _GaussianMultinomialDiffusion(
                k, x_num_t.shape[1], denoise, self.num_timesteps, self.device
            )
            diffusion.to(self.device)
        x = torch.from_numpy(np.concatenate([x_num_t, x_cat.astype(np.float32)], axis=1))
        n = len(x)
        opt = torch.optim.AdamW(diffusion.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        log(
            f"TabDDPM fit: {n:,} rows, n_iter={self.n_iter}, T={self.num_timesteps}, "
            f"batch={self.batch_size}, device={self.device}, KMP_DUPLICATE_LIB_OK="
            f"{os.environ.get('KMP_DUPLICATE_LIB_OK')}"
        )
        diffusion.train()
        for step in range(self.n_iter):
            frac = step / max(self.n_iter, 1)
            for group in opt.param_groups:
                group["lr"] = self.lr * (1.0 - frac)
            idx = torch.randint(0, n, (min(self.batch_size, n),))
            batch = x[idx].to(self.device)
            opt.zero_grad(set_to_none=True)
            loss_m, loss_g = diffusion.mixed_loss(batch)
            (loss_m + loss_g).backward()
            opt.step()
            if (step + 1) % 500 == 0 or step == 0:
                log(
                    f"TabDDPM step {step + 1}/{self.n_iter}  "
                    f"mloss={float(loss_m.detach()):.4f} gloss={float(loss_g.detach()):.4f}"
                )
        self._diffusion = diffusion
        log("TabDDPM fit finished")

    def sample(self, n: int) -> pd.DataFrame:
        if self._diffusion is None or self._qt is None:
            raise RuntimeError("call fit() before sample()")
        log(f"TabDDPM sample n={n:,} T={self.num_timesteps} batch={self.sample_batch}")
        chunks = []
        remaining = n
        while remaining > 0:
            b = min(self.sample_batch, remaining)
            chunks.append(self._diffusion._sample_batch(b))
            remaining -= b
            log(f"  sampled {n - remaining:,}/{n:,}")
        raw = torch.cat(chunks, dim=0)[:n].numpy()
        n_num = len(self.schema.numeric)
        x_num = self._qt.inverse_transform(raw[:, :n_num])
        if self._num_mins is not None and self._num_maxs is not None:
            # Keep heavy tails, but kill NaNs from quantile extremes.
            np.nan_to_num(x_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        for j in self._round_cols:
            x_num[:, j] = np.rint(x_num[:, j])
        cat_cols = list(self.schema.categorical) + [self.schema.label]
        data: dict[str, object] = {}
        for i, col in enumerate(self.schema.numeric):
            data[col] = x_num[:, i]
        x_cat = raw[:, n_num:].astype(int)
        for i, col in enumerate(cat_cols):
            levels = self._cat_levels[col]
            codes = np.clip(x_cat[:, i], 0, len(levels) - 1)
            vals = np.array(levels, dtype=object)[codes]
            data[col] = vals
        out = pd.DataFrame(data)
        out[self.schema.label] = out[self.schema.label].astype(int)
        return out[self.schema.all_cols]
