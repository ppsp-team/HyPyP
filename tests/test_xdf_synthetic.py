"""XDF import on synthetic files whose timing is known exactly.

Two EEG streams at 100 Hz, the second starting half a second after the
first, and two marker streams. Every expected value follows from how the
file was written, not from HyPyP.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hypyp.xdf import XDFImport
from hypyp.xdf.xdf_stream import Markers
from tests.xdf_writer import write_xdf

T0 = 1000.0
SRATE = 100.0
CLOCK = [(T0 + 5.0 * k, 0.0) for k in range(4)]


def eeg_stream(stream_id, name, t_start, n_samples, amplitude_uv, seed):
    rng = np.random.default_rng(seed)
    timestamps = t_start + np.arange(n_samples) / SRATE
    values = rng.normal(0.0, amplitude_uv, (n_samples, 2))
    return dict(
        id=stream_id,
        name=name,
        type="EEG",
        srate=SRATE,
        labels=["Ch1", "Ch2"],
        timestamps=timestamps,
        values=values,
        clock=CLOCK,
    )


def marker_stream(stream_id, name, times, descriptions):
    return dict(
        id=stream_id,
        name=name,
        type="Markers",
        srate=0,
        nch=1,
        fmt="string",
        timestamps=np.array(times),
        values=[[d] for d in descriptions],
        clock=CLOCK,
    )


@pytest.fixture
def dyad_file(tmp_path):
    return write_xdf(
        tmp_path / "dyad.xdf",
        [
            eeg_stream(1, "EEG1", T0, 1000, 20.0, seed=0),
            eeg_stream(2, "EEG2", T0 + 0.5, 1000, 20.0, seed=1),
            marker_stream(3, "Markers1", [T0 + 1.0], ["m1"]),
            marker_stream(4, "Markers2", [T0 + 2.5], ["m2"]),
        ],
    )


def test_markers_add_keeps_both_sets_of_timestamps_in_time_order():
    merged = Markers(np.array([1.0, 3.0]), ["a", "c"]) + Markers(np.array([2.0]), ["b"])
    assert_allclose(merged.timestamps, [1.0, 2.0, 3.0])
    assert merged.descriptions == ["a", "b", "c"]


def test_as_mne_annotations_sorts_and_drops_markers_before_the_reference():
    annotations = Markers(
        np.array([5.0, 1.0, 6.0]), ["c", "a", "d"]
    ).as_mne_annotations(2.0)
    assert_allclose(annotations.onset, [3.0, 4.0])
    assert list(annotations.description) == ["c", "d"]


def test_two_marker_streams_merge_with_their_own_timestamps(dyad_file):
    markers = XDFImport(dyad_file).markers
    assert_allclose(markers.timestamps, [T0 + 1.0, T0 + 2.5], atol=1e-3)
    assert markers.descriptions == ["m1", "m2"]


def test_every_signal_raw_receives_the_markers_relative_to_its_own_start(dyad_file):
    raws = XDFImport(dyad_file).mne_raws_dict
    assert_allclose(raws["EEG1"].annotations.onset, [1.0, 2.5], atol=1e-2)
    assert_allclose(raws["EEG2"].annotations.onset, [0.5, 2.0], atol=1e-2)
    assert list(raws["EEG1"].annotations.description) == ["m1", "m2"]


def test_scale_auto_converts_microvolts_to_volts(dyad_file):
    raw = XDFImport(dyad_file, select_matches=["EEG1"], scale="auto").mne_raws[0]
    assert raw.get_data().std() == pytest.approx(20e-6, rel=0.1)
