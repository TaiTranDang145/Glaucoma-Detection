"""Sampling helpers for multi-domain training."""

from collections import Counter
from typing import Iterable

import torch
from torch.utils.data import WeightedRandomSampler


def make_domain_balanced_sampler(domains: Iterable[str]) -> WeightedRandomSampler:
    """Sample training images with equal expected contribution from each domain.

    Each image receives weight ``1 / number_of_images_in_its_domain``. Thus,
    every domain has the same total sampling weight, regardless of its size.
    Sampling is with replacement and produces one epoch's worth of samples.
    """
    domain_names = list(domains)
    if not domain_names:
        raise ValueError("Cannot build a domain sampler from an empty dataset.")
    if any(not isinstance(domain, str) or not domain for domain in domain_names):
        raise ValueError("Every training sample must have a non-empty domain name.")

    counts = Counter(domain_names)
    weights = torch.tensor(
        [1.0 / counts[domain] for domain in domain_names],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(domain_names),
        replacement=True,
    )
