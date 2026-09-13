"""Empirical distribution objectives with analytic IQP circuit gradients.

Probabilities are the exact Born law: these objectives introduce neither a
probability floor nor a pseudocount. Logarithms use natural units. Existing
parity and MSE kernels are delegated unchanged so their trajectories remain
compatible with the original experiments.
"""
from __future__ import annotations

import numpy as np

from . import core


OBJECTIVES = ("parity", "mse", "scaled-mse", "nll", "spherical", "hellinger", "js", "tv")


def loss_gradient(circuit, theta, empirical, objective, weights=None):
    """Return ``(loss, dloss/dtheta)`` for a ``loss_search.Circuit``.

    ``empirical`` is the full ``2**n``-entry empirical probability vector and
    ``weights`` is required only for parity. MSE is the original support-mean
    squared error; scaled-MSE is its sum version. The additional losses are
    empirical NLL, spherical score ``1 - p.q / ||q||_2``, squared Hellinger
    distance, Jensen--Shannon divergence, and total variation distance.

    Entries with p=q=0 cause no singularities. NLL raises at an observed exact
    zero probability. Hellinger also raises if p>0 and q=0 because its circuit
    derivative can be undefined there. Jensen--Shannon has a zero amplitude
    derivative at q=0; TV uses the zero subgradient at q=p.
    """
    if objective in {"parity", "mse", "scaled-mse"}:
        return circuit.loss_gradient(theta, empirical, objective, weights)
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    empirical = np.asarray(empirical, dtype=float)
    if (empirical.shape != (circuit.size,) or not np.isfinite(empirical).all()
            or np.any(empirical < 0)):
        raise ValueError("empirical must be a finite, nonnegative full-state probability vector")

    diagonal, amplitude = circuit.state(theta)
    q = np.abs(amplitude)**2
    positive_p = empirical > 0
    positive_q = q > 0
    dq = np.zeros_like(q)

    if objective == "nll":
        if np.any(positive_p & ~positive_q):
            raise FloatingPointError("NLL is infinite at an observed exact zero probability")
        value = -np.dot(empirical[positive_p], np.log(q[positive_p]))
        dq[positive_p] = -empirical[positive_p] / q[positive_p]
    elif objective == "spherical":
        norm = np.linalg.norm(q)
        if norm == 0:
            raise FloatingPointError("spherical score requires a nonzero Born law")
        overlap = np.dot(empirical, q)
        value = 1 - overlap / norm
        dq = -empirical / norm + overlap * q / norm**3
    elif objective == "hellinger":
        if np.any(positive_p & ~positive_q):
            raise FloatingPointError("Hellinger circuit gradient is undefined at an observed exact zero probability")
        root_p, root_q = np.sqrt(empirical), np.sqrt(q)
        value = .5 * np.dot(root_p - root_q, root_p - root_q)
        # At p=q=0 the contribution is q/2, hence derivative 1/2.
        dq.fill(.5)
        dq[positive_p] -= .5 * root_p[positive_p] / root_q[positive_p]
    elif objective == "js":
        midpoint = .5 * (empirical + q)
        value = .5 * np.dot(empirical[positive_p],
                            np.log(empirical[positive_p] / midpoint[positive_p]))
        value += .5 * np.dot(q[positive_q], np.log(q[positive_q] / midpoint[positive_q]))
        dq[positive_q] = .5 * np.log(q[positive_q] / midpoint[positive_q])
        # dJ/dq diverges when q=0<p, but its amplitude derivative is zero:
        # lim_{a->0} a*log(|a|**2) = 0. Leave those entries at zero.
    else:  # total variation
        difference = q - empirical
        value = .5 * np.abs(difference).sum()
        dq = .5 * np.sign(difference)

    reverse = core.fwht(2 * dq * amplitude) / np.sqrt(circuit.size)
    phase_derivative = .5 * np.imag(np.conj(reverse) * diagonal)
    gradient = core.fwht(phase_derivative)[circuit.indices]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError(f"nonfinite {objective} loss or circuit gradient")
    return float(value), gradient
