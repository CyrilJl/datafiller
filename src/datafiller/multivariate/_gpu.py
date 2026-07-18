"""Optional GPU backend for the default-ridge Gram fast path.

PyTorch is imported lazily so the dependency stays optional: the module can
be imported without torch installed, and a helpful error is raised only when
a GPU device is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TORCH_INSTALL_HINT = (
    "GPU-accelerated imputation requires PyTorch, which is an optional dependency. "
    "Install it with `pip install datafiller[gpu]`, or pick a build matching your "
    "CUDA setup at https://pytorch.org/get-started/locally/. "
    "Leave `device=None` to use the default CPU implementation."
)


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_INSTALL_HINT) from exc
    return torch


@dataclass
class GpuColumnResult:
    """Outcome of one column's batched solve.

    `patterns`, `indexes` and `pattern_valid` are only materialized when at
    least one pattern could not be solved (``all_valid`` is False), so the
    caller can run the CPU fallback on those patterns.
    """

    predictions: np.ndarray  # (n_pred,) float32
    row_valid: np.ndarray  # (n_pred,) bool, prediction rows whose pattern was solved
    all_valid: bool
    patterns: np.ndarray | None = None  # (P, k) bool
    indexes: np.ndarray | None = None  # (n_pred,) pattern id of each prediction row
    pattern_valid: np.ndarray | None = None  # (P,) bool


class GramBackend:
    """Solves the per-pattern ridge systems of one column as batched tensor ops.

    Mirrors the CPU Gram fast path exactly: for each missingness pattern of
    the prediction rows, the ridge model is solved from the Gram matrix of
    the augmented training matrix ``[X, y, 1]`` accumulated over the rows
    that are complete on the pattern's usable columns. Instead of a Python
    loop over patterns, the backend computes

    1. usable rows for every pattern with one ``nan_mask @ patterns.T`` GEMM,
    2. all pattern Grams with one flat GEMM ``W.T @ (z ⊗ z)`` over row-wise
       outer products,
    3. all ridge coefficients with one batched ``linalg.solve`` where excluded
       columns are identity-padded (their coefficients are exactly zero).

    The input matrix is uploaded once per imputation call and column subsets
    are gathered on the device.
    """

    # Upper bounds on the (rows, K*K) outer-product buffer and on the
    # (rows, patterns) weight buffer; both are chunked past these sizes.
    _OUTER_BUDGET_BYTES = 256e6
    _WEIGHT_BUDGET_BYTES = 64e6

    def __init__(self, device):
        torch = _load_torch()
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device!r} was requested but torch.cuda.is_available() is False. "
                "Check that this PyTorch build supports CUDA and that a GPU driver is "
                "installed, or leave `device=None` to use the CPU implementation."
            )
        self._x_key = None
        self._x_gpu = None

    def release(self) -> None:
        """Drop the cached device copy of the input matrix."""
        self._x_key = None
        self._x_gpu = None

    def _bind(self, x: np.ndarray):
        """Upload ``x`` to the device once per imputation call."""
        if self._x_key != id(x):
            torch = self._torch
            host = np.ascontiguousarray(x, dtype=np.float32)
            self._x_gpu = torch.from_numpy(host).to(self.device)
            self._x_key = id(x)
        return self._x_gpu

    def _batched_grams(self, z, nan_train_f, patf):
        """Per-pattern Gram matrices, usable-row counts and validity masks."""
        torch = self._torch
        m_local, K = z.shape
        P = patf.shape[0]
        G = torch.empty((P, K * K), device=self.device, dtype=z.dtype)
        n_samp = torch.empty(P, device=self.device, dtype=z.dtype)
        pattern_chunk = max(64, int(self._WEIGHT_BUDGET_BYTES / (4 * m_local)))
        row_chunk = max(1024, int(self._OUTER_BUDGET_BYTES / (K * K * 4)))
        for ps in range(0, P, pattern_chunk):
            pe = min(P, ps + pattern_chunk)
            hits = nan_train_f @ patf[ps:pe].T  # (m, c)
            W = (hits == 0).to(z.dtype)
            n_samp[ps:pe] = W.sum(dim=0)
            if m_local <= row_chunk:
                Y = (z.unsqueeze(2) * z.unsqueeze(1)).reshape(m_local, K * K)
                G[ps:pe] = W.T @ Y
            else:
                G[ps:pe] = 0.0
                for rs in range(0, m_local, row_chunk):
                    re = min(m_local, rs + row_chunk)
                    zc = z[rs:re]
                    Y = (zc.unsqueeze(2) * zc.unsqueeze(1)).reshape(re - rs, K * K)
                    G[ps:pe] += W[rs:re].T @ Y
        return G.reshape(P, K, K), n_samp

    def impute_column(
        self,
        x: np.ndarray,
        col: int,
        trainable_rows: np.ndarray,
        imputable_rows: np.ndarray,
        sampled_cols: np.ndarray,
        alpha: float,
        fit_intercept: bool,
        min_samples_train: int,
    ) -> GpuColumnResult:
        """Solve all missingness patterns of one column in a batched pass."""
        torch = self._torch
        dev = self.device
        xg = self._bind(x)

        tr = torch.from_numpy(trainable_rows.astype(np.int64)).to(dev)
        im = torch.from_numpy(imputable_rows.astype(np.int64)).to(dev)
        sc = torch.from_numpy(sampled_cols.astype(np.int64)).to(dev)

        cols_g = xg.index_select(1, sc)  # (n_rows, k)
        local_train = cols_g.index_select(0, tr)  # (m, k)
        local_predict = cols_g.index_select(0, im)  # (n_pred, k)
        local_target = xg[:, col].index_select(0, tr)  # (m,)

        m_local, k = local_train.shape

        nan_train = torch.isnan(local_train)
        pat_u8, idx = torch.unique((~torch.isnan(local_predict)).to(torch.uint8), dim=0, return_inverse=True)
        pat = pat_u8.bool()  # (P, k)

        z = torch.cat(
            [
                torch.nan_to_num(local_train),
                local_target.unsqueeze(1),
                torch.ones((m_local, 1), device=dev, dtype=local_train.dtype),
            ],
            dim=1,
        )  # (m, K), NaNs zero-filled: they only sit in columns excluded per pattern

        G, n_samp = self._batched_grams(z, nan_train.to(z.dtype), pat.to(z.dtype))
        valid = (n_samp >= float(min_samples_train)) & pat.any(dim=1)
        nsf = n_samp.clamp(min=1.0)

        # Batched equivalent of fit_ridge_from_gram, with excluded columns
        # identity-padded so their coefficients solve to exactly zero.
        Sxx = G[:, :k, :k]
        sxy = G[:, :k, k]
        sx = G[:, :k, k + 1]
        sy = G[:, k, k + 1]
        if fit_intercept:
            A = Sxx - sx.unsqueeze(2) * sx.unsqueeze(1) / nsf.view(-1, 1, 1)
            b = sxy - sx * (sy / nsf).unsqueeze(1)
        else:
            A = Sxx
            b = sxy
        usable_pair = pat.unsqueeze(2) & pat.unsqueeze(1)
        A = A * usable_pair
        A = A + torch.diag_embed(torch.where(pat, alpha, 1.0))
        b = b * pat
        coef = torch.linalg.solve(A, b)  # (P, k)
        if fit_intercept:
            intercept = sy / nsf - (sx / nsf.unsqueeze(1) * coef).sum(dim=1)
        else:
            intercept = torch.zeros_like(sy)

        zp = torch.nan_to_num(local_predict)
        preds = (zp * coef.index_select(0, idx)).sum(dim=1) + intercept.index_select(0, idx)
        row_valid = valid.index_select(0, idx)

        predictions = preds.cpu().numpy()
        row_valid_np = row_valid.cpu().numpy()
        pattern_valid = valid.cpu().numpy()
        if pattern_valid.all():
            return GpuColumnResult(predictions=predictions, row_valid=row_valid_np, all_valid=True)
        return GpuColumnResult(
            predictions=predictions,
            row_valid=row_valid_np,
            all_valid=False,
            patterns=pat.cpu().numpy(),
            indexes=idx.cpu().numpy(),
            pattern_valid=pattern_valid,
        )
