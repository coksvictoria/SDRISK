"""Package-level smoke tests for the M0 scaffold."""

import sdrisk


def test_package_exposes_version() -> None:
    """The package can be imported after installation."""
    assert sdrisk.__version__ == "0.0.0"
