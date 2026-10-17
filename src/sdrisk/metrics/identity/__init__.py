"""Table-level identity disclosure metrics."""

from sdrisk.metrics.identity.dcr import dcr_overfitting, dcr_vs_random
from sdrisk.metrics.identity.exact_match import (
    exact_match_rate,
    unique_exact_match_rate,
)
from sdrisk.metrics.identity.kprofile import k_profile
from sdrisk.metrics.identity.repu import replicated_uniques

__all__ = [
    "dcr_overfitting",
    "dcr_vs_random",
    "exact_match_rate",
    "k_profile",
    "replicated_uniques",
    "unique_exact_match_rate",
]
