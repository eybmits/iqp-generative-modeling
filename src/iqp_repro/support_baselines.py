"""Classical extensions trained with the IQP benchmark's known even support.

Historical full-cube trainers remain unchanged. ``seed`` is the complete
initialization seed: this module applies no experiment-specific seed offset.
All learned models return the final iterate after ``config['steps']`` updates.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import classical, core


def edge_list(n, architecture):
    """The same pair-gate graphs as ring, extended ring, and dense IQP."""
    if architecture == 'dense':
        return list(itertools.combinations(range(n), 2))
    if architecture not in {'ring', 'ring3'}:
        raise ValueError('architecture must be ring, ring3, or dense')
    distances = (1, 2) if architecture == 'ring' else (1, 2, 3)
    return sorted({tuple(sorted((i, (i+d) % n)))
                   for i in range(n) for d in distances if i != (i+d) % n})


def _energy(theta, indices, size):
    coefficients = np.zeros(size, dtype=float)
    np.add.at(coefficients, indices, theta)
    return core.fwht(coefficients)


def _supported_distribution(logits, support):
    """Normalize only allowed states; retain exact zero mass elsewhere."""
    restricted = np.asarray(logits, dtype=float)[support]
    shifted = restricted - restricted.max()
    logq = np.full(len(support), -np.inf)
    logq[support] = shifted - np.log(np.exp(shifted).sum())
    return np.exp(logq), logq


def _ising_evaluate(theta, features, empirical, indices, support, loss, l2=0.):
    """Restricted energy-model objective and its exact covariance gradient."""
    q, logq = _supported_distribution(_energy(theta, features, len(support)), support)
    model_features = core.fwht(q)[features]
    if loss == 'nll':
        positive = empirical > 0
        value = -float(empirical[positive] @ logq[positive])
        gradient = model_features - core.fwht(empirical)[features]
    elif loss == 'parity':
        error = core.fwht(q-empirical)[indices]
        value = float(error @ error / len(indices))
        derivative = _energy(2*error/len(indices), indices, len(support))
        gradient = core.fwht(q*derivative)[features] - model_features*(q @ derivative)
    else:
        raise ValueError('Ising loss must be parity or nll')
    return (value + .5*l2*float(theta @ theta), gradient + l2*theta, q, logq)


def _train_ising(config, bits, empirical, masks, seed):
    n = bits.shape[1]
    loss = 'parity' if config['model'] == 'ising-parity' else 'nll'
    architecture = config.get('architecture', 'ring') if loss == 'parity' else 'dense'
    fields = [1 << (n-1-i) for i in range(n)]
    features = np.array([fields[i] | fields[j] for i,j in edge_list(n, architecture)] + fields)
    indices = core.mask_indices(masks)
    support = bits.sum(axis=1) % 2 == 0
    theta = .01*np.random.default_rng(seed).standard_normal(len(features))
    optimizer = core.Adam(lr=config['lr'])
    history = []
    for _ in range(config['steps']):
        value, gradient, _, _ = _ising_evaluate(
            theta, features, empirical, indices, support, loss, config['l2'])
        history.append(value)
        theta = optimizer.update(theta, gradient)
    value, _, q, logq = _ising_evaluate(
        theta, features, empirical, indices, support, loss, config['l2'])
    return dict(q=q, logq=logq, loss_history=np.array(history + [value]),
                parameters=len(theta), theta=theta, architecture=architecture)


def _embed(result, support, parameters):
    """Embed a normalized law on allowed states into the original cube."""
    logq = np.full(len(support), -np.inf)
    logq[support] = result['logq']
    return dict(q=np.exp(logq), logq=logq, loss_history=result['loss_history'],
                parameters=parameters)


def _train_maxent(config, bits, empirical, masks, seed):
    support = bits.sum(axis=1) % 2 == 0
    indices = core.mask_indices(masks)
    states = np.flatnonzero(support).astype(np.uint32)
    P = np.array([1-2*(np.bitwise_count(states & np.uint32(mask)) & 1).astype(np.float32)
                  for mask in indices], dtype=np.float32)
    moments = core.fwht(empirical)[indices]
    if config['l2'] == 0:
        # Matrix columns may be any finite state domain. The existing dense
        # trainer already computes its normalizer on exactly those columns.
        result = classical.train_maxent(P, moments, seed=seed,
                    steps=config['steps'], lr=config['lr'])
    else:
        theta = nn.Parameter(torch.zeros(len(indices), dtype=torch.float32))
        optimizer = torch.optim.Adam([theta], lr=config['lr'])
        features = torch.from_numpy(P)
        target = torch.as_tensor(moments, dtype=torch.float32)
        history = []
        for _ in range(config['steps']):
            optimizer.zero_grad(set_to_none=True)
            logits = theta @ features
            loss = (torch.logsumexp(logits, dim=0) - theta @ target
                    + .5*config['l2']*(theta @ theta))
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = theta @ features
            logz = torch.logsumexp(logits, dim=0)
            history.append((logz-theta @ target+.5*config['l2']*(theta @ theta)).item())
            logq = (logits-logz).numpy().astype(float)
            logq -= np.logaddexp.reduce(logq)
        result = dict(logq=logq, loss_history=np.array(history))
    return _embed(result, support, len(indices))


def _train_transformer(config, bits, samples, seed):
    n = bits.shape[1]
    prefixes = core.bits_table(n-1) if n > 2 else np.array([[0], [1]], dtype=np.int8)
    prefix_samples = samples >> 1
    kwargs = {name: config[name] for name in ('d_model', 'nhead', 'layers', 'dim_ff', 'batch_size')
              if name in config}
    if config['l2'] == 0:
        result = classical.train_transformer(prefixes, prefix_samples, seed=seed,
                    epochs=config['steps'], lr=config['lr'], **kwargs)
    else:
        torch.manual_seed(seed)
        x = torch.as_tensor(prefixes[prefix_samples], dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x),
            batch_size=kwargs.get('batch_size', 256), shuffle=True, drop_last=False,
            generator=torch.Generator().manual_seed(seed+11))
        model_kwargs = {k:v for k,v in kwargs.items() if k != 'batch_size'}
        model = classical.ARTransformer(n-1, **model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        history = []
        for _ in range(config['steps']):
            model.train()
            total = 0.
            for (xb,) in loader:
                data_loss = F.binary_cross_entropy_with_logits(
                    model(classical._input_tokens(xb)), xb.float())
                penalty = .5*config['l2']*sum((p*p).sum() for p in model.parameters())
                loss = data_loss + penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(xb)
            history.append(total/len(x))
        logq = classical.autoregressive_logq(model, prefixes)
        penalty = .5*config['l2']*sum(float((p.detach()*p.detach()).sum()) for p in model.parameters())
        history.append(float(-logq[prefix_samples].mean()/(n-1)) + penalty)
        logq -= np.logaddexp.reduce(logq)
        result = dict(logq=logq, loss_history=np.array(history))
    # Use fork_rng so counting model parameters does not change caller RNG state.
    with torch.random.fork_rng():
        model_kwargs = {k:v for k,v in kwargs.items() if k != 'batch_size'}
        parameter_count = sum(p.numel() for p in classical.ARTransformer(n-1, **model_kwargs).parameters())
    # In big-endian order each prefix has exactly one allowed final parity bit.
    support = bits.sum(axis=1) % 2 == 0
    return _embed(result, support, parameter_count)


def train_baseline(config, bits, samples, masks, seed):
    """Train a classical model on the known even support, without target access.

    Required config: ``model``, ``lr``, and ``steps``. Optional: ``l2`` (default
    zero), ``architecture`` for Ising parity, and Transformer capacity/batch
    arguments. L2 means half lambda times squared parameter norm, added to the
    training objective. MaxEnt retains duplicate sampled-feature parameters.
    ``support_aware`` defaults to True and cannot be False in this module.

    The optional ``spectral`` model uses the same empirical parity moments,
    clipped support-restricted inversion, then mixes ``alpha`` uniform mass
    (default .05). It has no optimized parameters and reports fitted_moments.
    """
    config = dict(config)
    if not config.get('support_aware', True):
        raise ValueError('support_baselines requires support_aware=True')
    if config.get('model') not in {'ising-parity', 'ising-nll', 'maxent', 'transformer', 'spectral'}:
        raise ValueError('unknown support-aware baseline model')
    config['l2'] = float(config.get('l2', 0.))
    config['lr'] = float(config.get('lr', .05))
    supplied_steps = config.get('steps', 600)
    config['steps'] = int(supplied_steps)
    if (config['steps'] != supplied_steps or config['steps'] < 0
            or not np.isfinite(config['lr']) or config['lr'] <= 0
            or not np.isfinite(config['l2']) or config['l2'] < 0):
        raise ValueError('require nonnegative integer steps, positive lr, and nonnegative l2')
    bits = np.asarray(bits)
    if bits.ndim != 2 or not np.array_equal(bits, core.bits_table(bits.shape[1])):
        raise ValueError('bits must enumerate the full cube in big-endian integer order')
    samples = np.asarray(samples)
    empirical = core.empirical(samples, bits.shape[1])
    support = bits.sum(axis=1) % 2 == 0
    if np.any(empirical[~support] > 0):
        raise ValueError('training samples must have even parity')
    masks = np.asarray(masks)
    if masks.ndim != 2 or masks.shape[1] != bits.shape[1] or len(masks) == 0:
        raise ValueError('masks must be a nonempty K by n binary array')
    core.mask_indices(masks)
    if config['model'].startswith('ising-'):
        return _train_ising(config, bits, empirical, masks, seed)
    if config['model'] == 'maxent':
        return _train_maxent(config, bits, empirical, masks, seed)
    if config['model'] == 'transformer':
        return _train_transformer(config, bits, samples, seed)
    alpha = float(config.get('alpha', .05))
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('spectral alpha must be between zero and one')
    q = (1-alpha)*core.spectral(empirical, masks, support) + alpha*support/support.sum()
    with np.errstate(divide='ignore'):
        logq = np.log(q)
    return dict(q=q, logq=logq, loss_history=np.array([]), parameters=0,
                fitted_moments=len(np.unique(core.mask_indices(masks))))
