"""Table-level attribute-disclosure metrics."""

from sdrisk.metrics.attribute.cap import cap_disclosure, tcap_disclosure
from sdrisk.metrics.attribute.disco import disco
from sdrisk.metrics.attribute.homogeneity import homogeneity
from sdrisk.metrics.attribute.ml_inference import ml_inference_gain
from sdrisk.metrics.attribute.numeric_tolerance import numeric_tolerance_recovery

__all__ = [
    "cap_disclosure",
    "disco",
    "homogeneity",
    "ml_inference_gain",
    "numeric_tolerance_recovery",
    "tcap_disclosure",
]
