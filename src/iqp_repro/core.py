"""Exact Boolean-cube benchmark, IQP circuit, Walsh loss and diagnostics.

Arrays use big-endian bit order. Simulation is exponential in n; Walsh transforms
avoid constructing the K by 2**n feature matrix. No quantum service is required.
"""

import numpy as np


def bits_table(n):
    """Row x contains the n-bit representation of integer x."""
    if not isinstance(n, (int, np.integer)) or not 2 <= n <= 24:
        raise ValueError("Exact enumeration requires an integer 2 <= n <= 24")
    return ((np.arange(2**n, dtype=np.uint32)[:, None]
             >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)


def pairs(n):
    """Sorted, unique nearest and next-nearest neighbor edges on a ring."""
    if n < 2:
        raise ValueError("n must be at least 2")
    return sorted({tuple(sorted((i, (i + d) % n)))
                   for i in range(n) for d in (1, 2) if i != (i + d) % n})


def target(n, beta):
    """p(x) proportional to exp(beta * longest bracketed zero run), on even x."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    bits = bits_table(n)
    support = bits.sum(axis=1) % 2 == 0
    last = np.full(len(bits), -1, dtype=np.int16)
    scores = np.zeros(len(bits), dtype=np.int16)
    for i in range(n):
        one = bits[:, i] == 1
        scores = np.maximum(scores, np.where(one & (last >= 0), i - last - 1, 0))
        last[one] = i
    p = np.zeros(len(bits))
    logits = beta * scores[support]
    p[support] = np.exp(logits - logits.max())
    p /= p.sum()
    return p, support, scores


def empirical(samples, n):
    """Training histogram; repeated observations retain their multiplicity."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or not samples.size or samples.dtype.kind not in "iu":
        raise ValueError("samples must be a nonempty vector of integer state IDs")
    if samples.min() < 0 or samples.max() >= 2**n:
        raise ValueError("sample outside the Boolean cube")
    return np.bincount(samples, minlength=2**n) / samples.size


def sample_masks(n, sigma, k, seed):
    """IID Bernoulli masks conditioned on nonzero, retaining duplicates.

    Rejection and NumPy RNG order match upstream experiments 1 and 2. Replacing
    an all-zero draw by a random singleton would change the mask distribution.
    """
    if n < 1 or k < 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("n, k and finite sigma must be positive")
    prob = -0.5 * np.expm1(-1.0 / (2.0 * sigma**2))
    if prob == 0:
        raise ValueError("sigma is too large for nonzero-mask rejection sampling")
    rng = np.random.default_rng(seed)
    masks = rng.binomial(1, prob, size=(k, n)).astype(np.int8)
    zero = np.flatnonzero(masks.sum(axis=1) == 0)
    while zero.size:
        masks[zero] = rng.binomial(1, prob, size=(zero.size, n))
        zero = np.flatnonzero(masks.sum(axis=1) == 0)
    return masks


def mask_indices(masks):
    masks = np.asarray(masks)
    if masks.ndim != 2 or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must be a binary K by n array")
    return masks.astype(np.int64) @ (1 << np.arange(masks.shape[1] - 1, -1, -1))


def fwht(values):
    """Unnormalized Walsh-Hadamard transform; preserves float/complex dtype."""
    result = np.array(values, copy=True)
    if result.ndim != 1 or not result.size or result.size & (result.size - 1):
        raise ValueError("Walsh input must be a vector of power-of-two length")
    if result.dtype.kind not in "fc":
        result = result.astype(np.float64)
    width = 1
    while width < result.size:
        block = result.reshape(-1, 2 * width)
        left, right = block[:, :width].copy(), block[:, width:].copy()
        block[:, :width], block[:, width:] = left + right, left - right
        width *= 2
    return result


class Adam:
    """PennyLane Adam: beta2=.99, epsilon before second-moment correction."""

    def __init__(self, lr=0.05, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t, self.m, self.v = 0, 0, 0

    def update(self, theta, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        rate = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        return theta - rate * self.m / (np.sqrt(self.v) + self.eps)


def _edge_indices(n):
    return np.array([(1 << (n - i - 1)) | (1 << (n - j - 1))
                     for i, j in pairs(n)])


def _state(n, theta):
    coefficients = np.zeros(2**n)
    coefficients[_edge_indices(n)] = theta
    # RZZ(theta) = exp(-i theta Z_i Z_j / 2); both H layers normalized.
    diagonal = np.exp(-0.5j * fwht(coefficients)) / np.sqrt(2**n)
    amplitude = fwht(diagonal) / np.sqrt(2**n)
    return diagonal, amplitude


def iqp_probabilities(n, theta):
    """One-layer H RZZ H Born probabilities in sorted ring-edge angle order."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(pairs(n)),) or not np.all(np.isfinite(theta)):
        raise ValueError("one finite angle per ring edge is required")
    _, amplitude = _state(n, theta)
    return np.abs(amplitude)**2


def iqp_loss_gradient(n, theta, emp, masks=None, loss="parity", mse_domain="support"):
    """Exact loss and reverse derivative; duplicates contribute separately."""
    diagonal, amplitude = _state(n, theta)
    q = np.abs(amplitude)**2
    if loss == "parity":
        indices = mask_indices(masks)
        error = fwht(q - emp)[indices]
        coefficients = np.zeros(q.size)
        np.add.at(coefficients, indices, 2 * error / len(indices))
        dq = fwht(coefficients)
        value = np.mean(error**2)
    elif loss == "mse":
        if mse_domain not in {"support", "cube"}:
            raise ValueError("mse_domain must be 'support' or 'cube'")
        # Both this single-layer IQP and the training data have exact even support.
        denominator = q.size // 2 if mse_domain == "support" else q.size
        difference = q - emp
        value = np.dot(difference, difference) / denominator
        dq = 2 * difference / denominator
    else:
        raise ValueError("loss must be 'parity' or 'mse'")
    reverse = fwht(2 * dq * amplitude) / np.sqrt(q.size)
    phase_derivative = 0.5 * np.imag(np.conj(reverse) * diagonal)
    gradient = fwht(phase_derivative)[_edge_indices(n)]
    return float(value), gradient


def train_iqp(n, emp, masks=None, steps=600, lr=0.05, seed_init=13695,
              loss="parity", mse_domain="support"):
    """Train exactly one IQP run. Final iterate, no best-step/oracle selection."""
    emp = _distribution(emp)
    if emp.size != 2**n or steps < 0 or lr <= 0:
        raise ValueError("invalid n, steps, lr or training histogram")
    odd = bits_table(n).sum(axis=1) % 2 == 1
    if np.any(emp[odd] > 0):
        raise ValueError("IQP benchmark training data must have even parity")
    if loss == "parity" and (masks is None or np.shape(masks)[1:] != (n,)):
        raise ValueError("parity loss requires K by n masks")
    theta = 0.01 * np.random.default_rng(seed_init).standard_normal(len(pairs(n)))
    optimizer = Adam(lr)
    history = []
    for _ in range(steps):
        value, gradient = iqp_loss_gradient(n, theta, emp, masks, loss, mse_domain)
        history.append(value)
        theta = optimizer.update(theta, gradient)
    history.append(iqp_loss_gradient(n, theta, emp, masks, loss, mse_domain)[0])
    q = iqp_probabilities(n, theta)
    q /= q.sum()
    return {"q": q, "theta": theta, "loss_history": np.array(history)}


def spectral(emp, masks, support=None):
    """Eq.13 sampled-band proxy, clipped and normalized (duplicates retained).

    support=None reproduces upstream full-cube normalization. Pass the even
    support mask for the support-restricted probability model described by Eq.14.
    """
    emp = _distribution(emp)
    indices = mask_indices(masks)
    coefficients = np.zeros(emp.size)
    coefficients[0] = 1
    np.add.at(coefficients, indices, fwht(emp)[indices])
    q = np.maximum(fwht(coefficients) / emp.size, 0)
    if support is not None:
        q[~np.asarray(support, dtype=bool)] = 0
    if q.sum() <= 0:
        raise ValueError("spectral projection has no positive mass on support")
    return q / q.sum()


def _distribution(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not p.size or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be a finite nonnegative vector")
    if not np.isclose(p.sum(), 1, rtol=1e-7, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return p


def forward_kl(p, q, eps=None):
    """True KL, including infinity. eps explicitly requests legacy smoothing.

    The upstream statistic floors *both* p and q then renormalizes, giving a
    finite number even when q misses positive target mass. It is not exact KL.
    """
    p, q = _distribution(p), _distribution(q)
    if p.shape != q.shape:
        raise ValueError("distributions must have equal shape")
    if eps is not None:
        if not 0 < eps < 1:
            raise ValueError("eps must lie between zero and one")
        p, q = np.maximum(p, eps), np.maximum(q, eps)
        p, q = p / p.sum(), q / q.sum()
    positive = p > 0
    if np.any(q[positive] == 0):
        return float("inf")
    return float(np.sum(p[positive] * (np.log(p[positive]) - np.log(q[positive]))))


def elite(scores, support, samples, tau=0.1, method="threshold"):
    """Unseen high-score states: paper threshold or original exact top-k cut.

    'threshold' includes all ties at the 1-tau quantile (paper Eq.9). 'topk'
    reproduces the source's argsort cut, whose tie order depends on NumPy.
    """
    if not 0 < tau <= 1:
        raise ValueError("tau must lie in (0, 1]")
    scores, support = np.asarray(scores), np.asarray(support, dtype=bool)
    if method == "threshold":
        selected = support & (scores >= np.quantile(scores[support], 1 - tau))
    elif method == "topk":
        valid = np.flatnonzero(support)
        count = max(1, int(np.floor(tau * len(valid))))
        selected = np.zeros_like(support)
        selected[valid[np.argsort(-scores[valid].astype(float))[:count]]] = True
    else:
        raise ValueError("method must be 'threshold' or 'topk'")
    selected[np.asarray(samples, dtype=int)] = False
    return selected


def coverage(q, elite, budgets):
    """Expected unique discoveries, fraction recovered, and discoveries/draw."""
    q = _distribution(q)
    probabilities = q[np.asarray(elite, dtype=bool)]
    budgets = np.atleast_1d(np.asarray(budgets, dtype=float))
    if not np.all(np.isfinite(budgets)) or np.any(budgets < 0):
        raise ValueError("budgets must be finite and nonnegative")
    discoveries = np.zeros(len(budgets))
    with np.errstate(divide="ignore"):
        log_miss = np.log1p(-probabilities)
    for i, budget in enumerate(budgets):
        if budget > 0:
            discoveries[i] = np.sum(-np.expm1(budget * log_miss))
    recovery = discoveries / len(probabilities) if len(probabilities) else np.full(len(budgets), np.nan)
    yield_ = np.divide(discoveries, budgets, out=np.full(len(budgets), np.nan), where=budgets > 0)
    return {"discoveries": discoveries, "recovery": recovery, "yield": yield_}
