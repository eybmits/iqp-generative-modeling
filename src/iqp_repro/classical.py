"""The four classical comparators in Appendix B, with exact normalization.

P may be a dense parity matrix or a vector of integer Walsh-mask indices.
The latter avoids storing K times 2**n features. Seeds are full initialization
seeds; the experiment driver applies the documented model-specific offsets.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import Adam, fwht, pairs


def _moments(q, P):
    return fwht(q)[P] if P.ndim == 1 else P @ q


def _logits(theta, P, size):
    if P.ndim == 2:
        return theta @ P
    coefficients = np.zeros(size, dtype=theta.dtype)
    np.add.at(coefficients, P, theta)  # Preserve duplicate sampled masks.
    return fwht(coefficients)


def _distribution(logits):
    shifted = logits - np.max(logits)
    logq = shifted - np.log(np.exp(shifted).sum())
    return np.exp(logq), logq


def _result(logq, history):
    # Double precision output retains a stable log probability even when exp
    # underflows. Renormalize only the final numerical roundoff, never support.
    logq = np.asarray(logq, dtype=np.float64)
    logq -= np.logaddexp.reduce(logq)
    return {"q": np.exp(logq), "logq": logq,
            "loss_history": np.asarray(history, dtype=np.float64)}


def train_maxent(P, z_data, *, seed=0, steps=600, lr=0.05, n=None, method="dense"):
    """Minimize log Z(theta) - theta.z, starting at zero (float32 Adam).

    ``n`` is required when P contains integer mask indices. The seed is accepted
    for the shared protocol; this zero-initialized deterministic model needs no
    random draws. The default preserves the upstream float32 matrix/autograd
    arithmetic (K*2**n*4 bytes, 2 GiB at n=20,K=512). Explicit method="walsh"
    saves memory but changes floating-point reductions and can change the final
    iterate appreciably after 600 Adam steps; it is a numerical variant.
    """
    P = np.asarray(P)
    size = 2 ** n if P.ndim == 1 else P.shape[1]
    if method not in {"dense", "walsh"}:
        raise ValueError("MaxEnt method must be dense or walsh")
    if method == "dense" and P.ndim == 1:
        states = np.arange(size, dtype=np.uint32)
        dense = np.empty((len(P), size), dtype=np.float32)
        for row, mask in enumerate(P):
            dense[row] = 1 - 2 * (np.bitwise_count(states & np.uint32(mask)) & 1).astype(np.float32)
        P = dense
    if P.ndim == 2:
        P = P.astype(np.float32, copy=False)
    z = np.asarray(z_data, dtype=np.float32)
    theta = nn.Parameter(torch.zeros(len(z), dtype=torch.float32))
    optimizer = torch.optim.Adam([theta], lr=lr)
    if method == "dense":
        P_t, z_t = torch.from_numpy(P), torch.from_numpy(z)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if method == "dense":
            logits = torch.matmul(theta, P_t)
            loss = torch.logsumexp(logits, dim=0) - torch.dot(theta, z_t)
            history.append(loss.item())
            loss.backward()
        else:
            weights = theta.detach().numpy()
            logits = _logits(weights, P, size)
            q, logq = _distribution(logits)
            history.append(float((logits - logq)[0] - weights @ z))
            gradient = _moments(q, P) - z
            theta.grad = torch.from_numpy(np.asarray(gradient, dtype=np.float32).copy())
        optimizer.step()
    if method == "dense":
        with torch.no_grad():
            logits = torch.matmul(theta, P_t)
            logz = torch.logsumexp(logits, dim=0)
            logq = (logits - logz).numpy()
            history.append((logz - torch.dot(theta, z_t)).item())
    else:
        weights = theta.detach().numpy()
        logits = _logits(weights, P, size)
        q, logq = _distribution(logits)
        history.append(float((logits - logq)[0] - weights @ z))
    return _result(logq, history)


def train_ising(bits, P, z_data, empirical, *, topology="nn_nnn",
                loss="parity", seed=0, steps=600, lr=0.05):
    """Pair couplings plus all local fields, float64 PennyLane-style Adam.

    Sparse NN+NNN uses parity MSE; the dense graph uses empirical NLL. Computing
    NLL through log probabilities avoids the upstream 1e-12 clipping plateau.
    """
    n = bits.shape[1]
    if topology not in {"nn_nnn", "dense"} or loss not in {"parity", "nll"}:
        raise ValueError("Choose topology nn_nnn/dense and loss parity/nll")
    edges = pairs(n) if topology == "nn_nnn" else [
        (i, j) for i in range(n) for j in range(i + 1, n)]
    fields = [1 << (n - 1 - i) for i in range(n)]
    features = np.asarray([fields[i] | fields[j] for i, j in edges] + fields)
    size = len(bits)
    P = np.asarray(P)
    z = np.asarray(z_data, dtype=np.float64)
    empirical = np.asarray(empirical, dtype=np.float64)
    empirical = empirical / empirical.sum()
    data_features = fwht(empirical)[features]
    theta = 0.01 * np.random.default_rng(seed).standard_normal(len(features))
    optimizer = Adam(lr=lr)
    history = []
    for _ in range(steps):
        q, logq = _distribution(_logits(theta, features, size))
        model_features = fwht(q)[features]
        if loss == "nll":
            history.append(float(-empirical @ logq))
            gradient = model_features - data_features
        else:
            error = _moments(q, P) - z
            history.append(float(error @ error / len(z)))
            derivative = _logits(2 * error / len(z), P, size)
            gradient = fwht(q * derivative)[features] - model_features * (q @ derivative)
        theta = optimizer.update(theta, gradient)
    q, logq = _distribution(_logits(theta, features, size))
    history.append(float(-empirical @ logq) if loss == "nll" else
                   float(np.mean((_moments(q, P) - z)**2)))
    return _result(logq, history)


class ARTransformer(nn.Module):
    """Big-endian next-bit model; 9,057 parameters at n=12 by default."""

    def __init__(self, n, d_model=32, nhead=4, layers=1, dim_ff=64):
        super().__init__()
        self.tok_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.0, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb[:, :length, :]
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        return self.out(self.encoder(x, mask=causal)).squeeze(-1)


def _input_tokens(bits):
    bos = torch.full((len(bits), 1), 2, dtype=torch.long, device=bits.device)
    return torch.cat([bos, bits[:, :-1]], dim=1)


def autoregressive_logq(model, bits, batch_size=2048):
    """Enumerate the chain rule without clipping or parity postselection.

    Returned values are the raw chain-rule log probabilities, before a possible
    floating-point normalization correction. Stable logsigmoid avoids turning
    confident but finite predictions into exact zeros or artificial floors.
    """
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(bits), batch_size):
            x = torch.as_tensor(bits[start:start + batch_size], dtype=torch.long)
            logits = model(_input_tokens(x)).double()
            chunks.append((-F.binary_cross_entropy_with_logits(
                logits, x.double(), reduction="none").sum(1)).numpy())
    return np.concatenate(chunks)


def train_transformer(bits, train_indices, *, seed=0, epochs=600, lr=0.001,
                      batch_size=256, d_model=32, nhead=4, layers=1, dim_ff=64):
    """Fixed-budget next-bit likelihood training with the upstream RNG order."""
    torch.manual_seed(seed)
    x = torch.as_tensor(bits[np.asarray(train_indices)], dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=batch_size, shuffle=True,
        drop_last=False, generator=torch.Generator().manual_seed(seed + 11))
    model = ARTransformer(bits.shape[1], d_model, nhead, layers, dim_ff)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for (xb,) in loader:
            loss = F.binary_cross_entropy_with_logits(model(_input_tokens(xb)), xb.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(x))
    logq = autoregressive_logq(model, bits)
    history.append(float(-logq[np.asarray(train_indices)].mean() / bits.shape[1]))
    return _result(logq, history)
