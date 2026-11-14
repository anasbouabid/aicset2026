"""FinDiff backbone (Sattarov et al., ICAIF 2023).

Gaussian DDPM on concatenated categorical embeddings plus quantile-normalized
numerics. Categoricals, including the label, are joint table columns (same as
CTGAN/TabDDPM) — not classifier-free label conditioning.

Architecture follows the published paper: linear schedule, T=500, embedding
dimension 2, timestep embedding 64, MLP residual backbone. Training uses a
matched step budget with TabDDPM (not the paper's 3000-epoch Credit-Default
search). Official ``sattarov/FinDiff`` is not pip-installed: it would add a
second torch stack. This module uses the torch already in ``synthfraud``.

On macOS, set ``KMP_DUPLICATE_LIB_OK=TRUE`` before importing torch.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from synthfraud.log import log
from synthfraud.schema import Schema


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _SinusoidalTime(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=time.device) * -freqs)
        args = time[:, None].float() * freqs[None]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer2(self.act(self.layer1(x)))


class _MLPBackbone(nn.Module):
    def __init__(self, num_features: int, cond_dim: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_features + cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, num_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(torch.cat([x, cond], dim=1))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class _FinDiffNet(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_sizes: list[int],
        cat_emb_dim: int,
        time_dim: int,
        hidden_dim: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.cat_emb_dim = cat_emb_dim
        self.n_cat = len(cat_sizes)
        self.dim_input = n_num + self.n_cat * cat_emb_dim
        self.cat_embs = nn.ModuleList([nn.Embedding(k, cat_emb_dim) for k in cat_sizes])
        self.time_mlp = nn.Sequential(
            _SinusoidalTime(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.backbone = _MLPBackbone(self.dim_input, time_dim, hidden_dim, num_blocks)

    def embed_cat(self, x_cat: torch.Tensor) -> torch.Tensor:
        pieces = [self.cat_embs[j](x_cat[:, j]) for j in range(self.n_cat)]
        return torch.cat(pieces, dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, self.time_mlp(t))


class _LinearDiffuser:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.steps = steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, steps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_hat = torch.cumprod(alphas, dim=0)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_ah = torch.sqrt(self.alphas_hat[t])[:, None]
        sqrt_om = torch.sqrt(1.0 - self.alphas_hat[t])[:, None]
        noise = torch.randn_like(x)
        return sqrt_ah * x + sqrt_om * noise, noise

    def p_sample(self, eps: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_a = torch.sqrt(self.alphas[t])[:, None]
        betas_t = self.betas[t][:, None]
        sqrt_om = torch.sqrt(1.0 - self.alphas_hat[t])[:, None]
        sigma = torch.sqrt(self.betas[t])[:, None]
        noise = torch.randn_like(z)
        noise = torch.where((t == 0)[:, None], torch.zeros_like(noise), noise)
        mean = (1.0 / sqrt_a) * (z - betas_t * eps / sqrt_om)
        return mean + sigma * noise


class FinDiffSynthesizer:
    def __init__(
        self,
        seed: int,
        schema: Schema,
        n_iter: int = 10000,
        num_timesteps: int = 500,
        batch_size: int = 1024,
        sample_batch: int = 4096,
        lr: float = 1e-4,
        cat_emb_dim: int = 2,
        time_dim: int = 64,
        hidden_dim: int = 1024,
        num_blocks: int = 4,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.seed = int(seed)
        self.schema = schema
        self.n_iter = int(n_iter)
        self.num_timesteps = int(num_timesteps)
        self.batch_size = int(batch_size)
        self.sample_batch = int(sample_batch)
        self.lr = float(lr)
        self.cat_emb_dim = int(cat_emb_dim)
        self.time_dim = int(time_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = _device()
        self._net: _FinDiffNet | None = None
        self._diffuser: _LinearDiffuser | None = None
        self._qt: QuantileTransformer | None = None
        self._cat_levels: dict[str, list[str]] = {}
        self._round_cols: list[int] = []
        self._num_mins: np.ndarray | None = None
        self._num_maxs: np.ndarray | None = None

    def _cat_cols(self) -> list[str]:
        return list(self.schema.categorical) + [self.schema.label]

    def _encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_num = frame[list(self.schema.numeric)].to_numpy(dtype=np.float64)
        codes = []
        for col in self._cat_cols():
            mapping = {v: i for i, v in enumerate(self._cat_levels[col])}
            codes.append(frame[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64))
        return x_num, np.stack(codes, axis=1)

    def _decode_cat(self, cat_emb: torch.Tensor) -> np.ndarray:
        assert self._net is not None
        b = cat_emb.shape[0]
        pieces = []
        offset = 0
        for j, emb in enumerate(self._net.cat_embs):
            chunk = cat_emb[:, offset : offset + self.cat_emb_dim]
            offset += self.cat_emb_dim
            table = emb.weight.detach()
            dist = torch.cdist(chunk, table)
            pieces.append(dist.argmin(dim=1))
        return torch.stack(pieces, dim=1).cpu().numpy() if pieces else np.zeros((b, 0), dtype=np.int64)

    def fit(self, train: pd.DataFrame) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        table = train[self.schema.all_cols].copy()
        table[self.schema.label] = table[self.schema.label].astype(int).astype(str)
        for col in self.schema.categorical:
            table[col] = table[col].astype(str)
        cat_cols = self._cat_cols()
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
        cat_sizes = [len(self._cat_levels[c]) for c in cat_cols]
        net = _FinDiffNet(
            n_num=x_num_t.shape[1],
            cat_sizes=cat_sizes,
            cat_emb_dim=self.cat_emb_dim,
            time_dim=self.time_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
        )
        try:
            net = net.to(self.device)
            _ = net(
                torch.zeros((2, net.dim_input), device=self.device),
                torch.zeros((2,), device=self.device, dtype=torch.long),
            )
        except Exception as exc:
            log(f"FinDiff device {self.device} failed ({exc!r}); falling back to CPU")
            self.device = torch.device("cpu")
            net = net.to(self.device)
        self._net = net
        self._diffuser = _LinearDiffuser(
            self.num_timesteps, self.beta_start, self.beta_end, self.device
        )
        x_num_t = torch.from_numpy(x_num_t)
        x_cat_t = torch.from_numpy(x_cat.astype(np.int64))
        n = len(x_num_t)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        log(
            f"FinDiff fit: {n:,} rows, n_iter={self.n_iter}, T={self.num_timesteps}, "
            f"batch={self.batch_size}, hidden={self.hidden_dim}x{self.num_blocks}, "
            f"device={self.device}, KMP_DUPLICATE_LIB_OK="
            f"{os.environ.get('KMP_DUPLICATE_LIB_OK')}"
        )
        net.train()
        for step in range(self.n_iter):
            frac = step / max(self.n_iter, 1)
            for group in opt.param_groups:
                group["lr"] = self.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
            idx = torch.randint(0, n, (min(self.batch_size, n),))
            batch_num = x_num_t[idx].to(self.device)
            batch_cat = x_cat_t[idx].to(self.device)
            x0 = torch.cat([net.embed_cat(batch_cat), batch_num], dim=1)
            t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=self.device)
            xt, noise = self._diffuser.add_noise(x0, t)
            pred = net(xt, t)
            loss = ((pred - noise) ** 2).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 500 == 0 or step == 0:
                log(f"FinDiff step {step + 1}/{self.n_iter}  loss={float(loss.detach()):.4f}")
        log("FinDiff fit finished")

    @torch.no_grad()
    def sample(self, n: int) -> pd.DataFrame:
        if self._net is None or self._diffuser is None or self._qt is None:
            raise RuntimeError("call fit() before sample()")
        log(f"FinDiff sample n={n:,} T={self.num_timesteps} batch={self.sample_batch}")
        self._net.eval()
        cat_dim = self._net.n_cat * self.cat_emb_dim
        chunks_num: list[np.ndarray] = []
        chunks_cat: list[np.ndarray] = []
        remaining = n
        while remaining > 0:
            b = min(self.sample_batch, remaining)
            z = torch.randn((b, self._net.dim_input), device=self.device)
            for i in reversed(range(self.num_timesteps)):
                t = torch.full((b,), i, device=self.device, dtype=torch.long)
                z = self._diffuser.p_sample(self._net(z.float(), t), z, t)
            z_np = z.cpu().numpy()
            chunks_cat.append(self._decode_cat(z[:, :cat_dim]))
            chunks_num.append(z_np[:, cat_dim:])
            remaining -= b
            log(f"  sampled {n - remaining:,}/{n:,}")
        x_num = self._qt.inverse_transform(np.concatenate(chunks_num, axis=0)[:n])
        np.nan_to_num(x_num, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        for j in self._round_cols:
            x_num[:, j] = np.rint(x_num[:, j])
        x_cat = np.concatenate(chunks_cat, axis=0)[:n]
        data: dict[str, object] = {}
        for i, col in enumerate(self.schema.numeric):
            data[col] = x_num[:, i]
        for i, col in enumerate(self._cat_cols()):
            levels = self._cat_levels[col]
            codes = np.clip(x_cat[:, i], 0, len(levels) - 1)
            data[col] = np.array(levels, dtype=object)[codes]
        out = pd.DataFrame(data)
        out[self.schema.label] = out[self.schema.label].astype(int)
        return out[self.schema.all_cols]
