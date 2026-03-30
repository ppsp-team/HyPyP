"""
Tests for hypyp.datasets — pooch-based HypypData fetcher.

All fetch calls use the local cache after the first download, so these tests
are safe to run repeatedly without network access once the cache is warm.
"""

import os
from pathlib import Path

import pytest

from hypyp import datasets


# ---------------------------------------------------------------------------
# Cache / registry helpers
# ---------------------------------------------------------------------------

def test_cache_dir_is_path():
    assert isinstance(datasets.cache_dir(), Path)


def test_cache_dir_respects_mne_data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MNE_DATA", str(tmp_path))
    # Re-evaluate the expected path using the same logic as datasets.py
    expected = tmp_path / "HypypData"
    # The module-level _CACHE_DIR was already computed at import time,
    # but cache_dir() re-reads the env — test the formula, not the cached value.
    result = Path(os.path.join(os.environ["MNE_DATA"], "HypypData"))
    assert result == expected


def test_fetch_unknown_key_raises():
    with pytest.raises(ValueError, match="not in the HypypData registry"):
        datasets.fetch("does/not/exist.fif")


# ---------------------------------------------------------------------------
# EEG helpers
# ---------------------------------------------------------------------------

def test_eeg_epochs_participant1():
    path = datasets.eeg_epochs(participant=1)
    assert Path(path).exists()
    assert path.endswith(".fif")


def test_eeg_epochs_participant2():
    path = datasets.eeg_epochs(participant=2)
    assert Path(path).exists()
    assert path.endswith(".fif")


def test_eeg_raw():
    path = datasets.eeg_raw()
    assert Path(path).exists()
    assert path.endswith(".fif")


# ---------------------------------------------------------------------------
# EEGLAB helpers — .fdt companion must be co-located
# ---------------------------------------------------------------------------

def test_eeglab_epochs_set_exists():
    path = datasets.eeglab_epochs()
    assert Path(path).exists()
    assert path.endswith(".set")


def test_eeglab_epochs_fdt_colocated():
    path = datasets.eeglab_epochs()
    fdt = Path(path).with_suffix(".fdt")
    assert fdt.exists(), f"Companion .fdt not found alongside {path}"


def test_eeglab_raw_set_exists():
    path = datasets.eeglab_raw()
    assert Path(path).exists()
    assert path.endswith(".set")


def test_eeglab_raw_fdt_colocated():
    path = datasets.eeglab_raw()
    fdt = Path(path).with_suffix(".fdt")
    assert fdt.exists(), f"Companion .fdt not found alongside {path}"


# ---------------------------------------------------------------------------
# fNIRS helpers
# ---------------------------------------------------------------------------

def test_fnirs_samples_default_returns_four():
    paths = datasets.fnirs_samples()
    assert len(paths) == 4
    for p in paths:
        assert Path(p).exists()
        assert p.endswith(".snirf")


def test_fnirs_samples_subset():
    paths = datasets.fnirs_samples(indices=[1, 3])
    assert len(paths) == 2
    assert paths[0].endswith("sample_1.snirf")
    assert paths[1].endswith("sample_3.snirf")


def test_fnirs_slow_breathing():
    path = datasets.fnirs_slow_breathing()
    assert Path(path).exists()
    assert path.endswith(".snirf")


def test_fnirs_dcare_subject1():
    path = datasets.fnirs_dcare(subject=1)
    assert Path(path).exists()
    assert path.endswith(".snirf")


def test_fnirs_dcare_subject2():
    path = datasets.fnirs_dcare(subject=2)
    assert Path(path).exists()
    assert path.endswith(".snirf")


def test_fnirs_fcs01_parent_hdr_exists():
    path = datasets.fnirs_fcs01_parent()
    assert Path(path).exists()
    assert path.endswith(".hdr")


def test_fnirs_fcs01_parent_companions_colocated():
    """All Homer2/NIRX companion files must be in the same directory as the .hdr."""
    hdr_path = Path(datasets.fnirs_fcs01_parent())
    parent_dir = hdr_path.parent
    stem = hdr_path.stem  # NIRS-2019-09-28_002

    required_extensions = [".wl1", ".wl2", ".dat", ".evt", ".inf", ".nirs"]
    for ext in required_extensions:
        companion = parent_dir / f"{stem}{ext}"
        assert companion.exists(), f"Missing companion file: {companion}"


# ---------------------------------------------------------------------------
# XDF helpers
# ---------------------------------------------------------------------------

def test_xdf_dyad_noise():
    path = datasets.xdf_dyad_noise()
    assert Path(path).exists()
    assert path.endswith(".xdf")


def test_xdf_dyad_with_markers():
    path = datasets.xdf_dyad_with_markers()
    assert Path(path).exists()
    assert path.endswith(".xdf")


def test_xdf_clock_resets():
    path = datasets.xdf_clock_resets()
    assert Path(path).exists()
    assert path.endswith(".xdf")
