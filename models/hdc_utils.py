# Kenny Schlegel, Dmitri A. Rachkovskij, Denis Kleyko, Ross W. Gayler, Peter Protzel, Peer Neubert
#
# Structured temporal representation in time series classification with ROCKETs and Hyperdimensional Computing
# Copyright (C) 2025 Chair of Automation Technology / TU Chemnitz

import numpy as np
from PIL.ImageChops import offset
from numpy.fft import fft, ifft, ifftshift


def fpe_sinusoid(scalar, phases, bandwidth, offset=1):
    """
    Sinusoidal fractional binding for scalar encoding.

    Args:
        inputs: Scalar input vector.
        phases: Seed vector [#dim].
        bandwidth: Scaling factor for similarity.
        offset: Offset used if scale is zero to prevent zeros in final encoding.

    Returns:
        Fourier feature encoding using sine and cosine.
    """
    exponent = bandwidth * scalar + offset
    output = phases * exponent
    c = np.cos(output)
    s = np.sin(output)[::-1]
    return np.concatenate((c, s), axis=0)


def get_phases(D, kernel, seed=0):
    np.random.seed(seed)

    if kernel == 'transformer':
        phases = 1./10000.**(2*np.arange(D) // D)
    elif kernel == 'sinc':
        phases = np.random.rand(D) * 2 * np.pi - np.pi
    elif kernel == 'gaussian':
        phases = np.random.normal(0, 1, D)
    elif kernel == 'triangular':
        p = np.power(np.sinc(np.linspace(-np.pi, np.pi, D)), 2)
        phases = np.random.choice(np.linspace(-np.pi, np.pi, D), D, p=p / p.sum())
    else:
        raise ValueError("Unknown kernel")

    if D % 2 == 1:
        half = phases[: (D - 1) // 2]
        phases = np.concatenate([half, [0], -np.flip(half)])
    else:
        half = phases[: (D - 2) // 2]
        phases = np.concatenate([[0], half, [0], -np.flip(half)])

    return phases

def make_base(phases):
    baseFFT = np.exp(1j * phases)
    return np.real(ifft(baseFFT))

def make_FPE(base, bandwidth, scalar):
    return np.real(ifft(fft(base) ** (bandwidth * scalar)))

def create_FPE(scalars, bandwidth, HDC_dim, seed=0, fpe_method='orig', kernel='sinc', scalar_offset=1):
    """
    Encode poses (e.g., timestamps or positions) into a position matrix.

    Args:
        scalars: Input scalar values to be encoded.
        bandwidth: Scale factor for fractional binding (similarity decrease).
        HDC_dim: Dimensionality of the HDC space.
        seed: Random seed for generating initial phases.
        fpe_method: Method for fractional binding ('orig', 'sinusoid', 'cosine', etc.).
        kernel: Kernel type for initialization ('sinc', 'gaussian', 'triangular').
        scalar_offset: Offset to prevent zeros in final encoding (default is 1).

    Returns:
        Pose matrix encoding the input scalars.
    """
    np.random.seed(seed)

    # Initialize phase vector based on kernel type
    phases = get_phases(HDC_dim, kernel, seed)
    phasor = phases[:HDC_dim // 2]

    # Apply the specified FPE method
    fpe_methods = {
        'sinusoid': lambda: np.array([fpe_sinusoid(s, phasor, bandwidth, offset=scalar_offset) for s in scalars]),
    }

    if fpe_method in fpe_methods:
        poses = fpe_methods[fpe_method]()
    else:
        raise ValueError('Invalid FPE method.')

    if fpe_method == "orig_hp":
        # Standardize the results
        poses = (poses.T - np.mean(poses, axis=1)).T / np.std(poses, axis=1, keepdims=True)

    return poses.astype(np.float32)