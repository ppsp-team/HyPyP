"""
hypyp.datasets — Automatic download and caching of HypypData sample files.

Uses pooch to fetch files from the HypypData GitHub repository and cache
them locally in ``~/mne_data/HypypData/`` (or ``$MNE_DATA/HypypData/``).
Files are only downloaded once; subsequent calls use the local cache.

Example
-------
>>> from hypyp import datasets
>>> epo1_path = datasets.eeg_epochs(participant=1)
>>> snirf_paths = datasets.fnirs_samples()
>>> xdf_path = datasets.xdf_dyad_noise()

References
----------
HypypData repository:
    https://github.com/ppsp-team/HypypData
pooch documentation:
    https://www.fatiando.org/pooch/
"""

from __future__ import annotations

import os
from pathlib import Path

import pooch

# ---------------------------------------------------------------------------
# Registry — SHA256 checksums for all sample files.
# These must be updated whenever files in HypypData are modified.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, str] = {
    # MNE .fif — epochs
    "mne/epochs/participant1-epo.fif": (
        "sha256:cb23ed7ca7db6ae221018f928834104d1140d88ca76e7ffe74fb1f5208cbc049"
    ),
    "mne/epochs/participant2-epo.fif": (
        "sha256:62f81ff5eca6be6bd6dd505f444e3bd9f8a1897653ab01c32db0626822128ed3"
    ),
    # MNE .fif — raw
    "mne/raw/sub-110_session-1_pre.fif": (
        "sha256:630e9ffa37b5f3ffd7eeaa7b99de5517d30137bc369e2cb58f5473687a3efa4d"
    ),
    # EEGLAB
    "eeglab/eeglab_data.set": (
        "sha256:27594dc378685d25d5f5f762f4f8332e7a3d988b6d5d41832bb2252d93988718"
    ),
    "eeglab/eeglab_data.fdt": (
        "sha256:3ec388b9a080c723c00cd380403d64cc754929d828375bbd53bd83b468136759"
    ),
    "eeglab/eeglab_data_epochs_ica.set": (
        "sha256:8aa47f3d4e6fb6aa3fa3120806e6017e7b652e92d62221fa7115a75e3ed8a699"
    ),
    "eeglab/eeglab_data_epochs_ica.fdt": (
        "sha256:466cbea87b0fb2298db291011902000848e10ec265ed982dd54a8fa768d5e11f"
    ),
    "eeglab/eeglab_chan32.locs": (
        "sha256:c74c38260064ae042f482c00d114cab85219bcdfe2639daf839252338417a264"
    ),
    # fNIRS — sample files (mono-subject recordings used as hyperscanning proxy)
    "fnirs/samples/sample_1.snirf": (
        "sha256:0e2f5ea94684431b6c1843a085575518b57b9e801650644f7040044cbe974b01"
    ),
    "fnirs/samples/sample_2.snirf": (
        "sha256:90334cb82be6b290a267e0f1064bd7ead31825cb2ab3568b73fffd11ee90cd50"
    ),
    "fnirs/samples/sample_3.snirf": (
        "sha256:9cfe7fc4e568b103ad59148f78ed20009a171fa60efc7eac14ba601051ffe54d"
    ),
    "fnirs/samples/sample_4.snirf": (
        "sha256:f4ef1600c6977ed3b38647ea23e09e880fdbf606657508f83b5bbeaa01c64ea6"
    ),
    "fnirs/samples/slow_breathing.snirf": (
        "sha256:56e9a60f7bc6bf6f6233a5b990cc0138db295560fcfa103cd6323f9367062a2f"
    ),
    # XDF
    "xdf/dyad-example-noise.xdf": (
        "sha256:fb3065b44d6d55844749dc0e101663766c7d20794daf2cef0587b9279bb47087"
    ),
    "xdf/dyad-example-with-markers.xdf": (
        "sha256:48820f4a822fedc581e7960196f0ccf57d1bf0345bada196019f5886c3414d27"
    ),
    "xdf/data_with_clock_resets.xdf": (
        "sha256:45843167eb52eb111e27aa1cb64f000f82d94598158150f962fad8250ad17d2f"
    ),
    # fNIRS — LIONirs reference files (channel-to-ROI mapping)
    "fnirs/lionirs/channel_grouping_7ROI.mat": (
        "sha256:bd7ebeec57ed7b51186f2b7f2a7f1e13bc6611676fb31e2e6ad6cb5cc92f5686"
    ),
    "fnirs/lionirs/Standard_Channels.txt": (
        "sha256:0e1089bea99c73989b5b0b9e2c322d28606fd14d405bdc5c16c927bf09eb6eba"
    ),
    # fNIRS — DCARE study (real hyperscanning dyad)
    "fnirs/DCARE/DCARE_02_sub1.snirf": (
        "sha256:171590fa06851e78575639705424d973d1874d99492909788569940b63d3ca5f"
    ),
    "fnirs/DCARE/DCARE_02_sub2.snirf": (
        "sha256:4b2fead7a704daae1c345a015845ee50ecd44bb16acb9bd632090c7611436248"
    ),
    "fnirs/DCARE/MCARE_01_probeInfo.mat": (
        "sha256:227dedbfdd7d73fe945adf8e9c4682526109e29b67c754f7b48ce95668f21435"
    ),
    # fNIRS — FCS01 raw recordings (Homer2/NIRX format, parent participant)
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.hdr": (
        "sha256:cae48a5c0b336e58b9b020625a97eb3f5923f0c1e63bca11c38c73427262cb77"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.wl1": (
        "sha256:6a0aadef656270b1c44157a2de48ff4a698068df448342f19b02767cc728bff9"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.wl2": (
        "sha256:df24789b494bfed753f9d896477356e6b46a5354e3a4e121b3854a47c4db6c51"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.dat": (
        "sha256:284198339be721aa4db9b0278340a702a686efabbaabe24630b7de2c78d317d9"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.evt": (
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.inf": (
        "sha256:a7e89de81c472b492fb8255d0d9529b95737bce8da7616cc0b47a58e55209c0c"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.nirs": (
        "sha256:ef595d110863cedeecbbc04cc25c519f2a72c07ef0198a57828f5536394a38f6"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.avg": (
        "sha256:a339f624945d96e8c7e3d6c6e9fa8c3a2ba23f7ff350e327a2ac5a01cea54283"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.set": (
        "sha256:8bfd9e5e6eb0b2aea1248cb1448bd77bb9177a806026c7dfcee8662100d181ce"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002.tpl": (
        "sha256:1aa246bd8a92b844a0489f9096963f2ee598b4a01390ba9c03c6ebf87c6f18b4"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002_config.txt": (
        "sha256:39a9599b22e729498785f2e6cf1c0104bb5ed49820ac3d16250b0008a91738ae"
    ),
    "fnirs/FCS01/parent/NIRS-2019-09-28_002_probeInfo.mat": (
        "sha256:778215fc1b5c819ce5433e45d62e6bc515c19ca5a606ca972cb50026f99c6449"
    ),
}

# ---------------------------------------------------------------------------
# Pooch fetcher setup
# ---------------------------------------------------------------------------

_BASE_URL = "https://raw.githubusercontent.com/ppsp-team/HypypData/main/"

_CACHE_DIR = os.path.join(
    os.environ.get("MNE_DATA", os.path.expanduser("~/mne_data")),
    "HypypData",
)

_fetcher = pooch.create(
    path=_CACHE_DIR,
    base_url=_BASE_URL,
    registry=_REGISTRY,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch(filename: str, force_download: bool = False) -> str:
    """
    Fetch a file from HypypData by its registry key, caching locally.

    Downloads the file from GitHub on the first call and returns the local
    cached path on subsequent calls. The cache lives in
    ``~/mne_data/HypypData/`` (or ``$MNE_DATA/HypypData/``).

    Parameters
    ----------
    filename : str
        Path relative to the HypypData repository root, e.g.
        ``"mne/epochs/participant1-epo.fif"``.
    force_download : bool
        If True, re-download even if the file is already cached.
        Default: False.

    Returns
    -------
    str
        Absolute path to the locally cached file.

    Raises
    ------
    ValueError
        If ``filename`` is not in the registry.

    Example
    -------
    >>> path = fetch("mne/epochs/participant1-epo.fif")
    >>> print(path)  # ~/.mne_data/HypypData/mne/epochs/participant1-epo.fif
    """
    if filename not in _REGISTRY:
        raise ValueError(
            f"'{filename}' is not in the HypypData registry. "
            f"Available files:\n  " + "\n  ".join(sorted(_REGISTRY))
        )
    return _fetcher.fetch(filename)


def eeg_epochs(participant: int = 1) -> str:
    """
    Fetch preprocessed EEG epochs for a given participant.

    Parameters
    ----------
    participant : int
        Participant index (1 or 2). Default: 1.

    Returns
    -------
    str
        Local path to the ``participant{N}-epo.fif`` file.

    Example
    -------
    >>> import mne
    >>> from hypyp import datasets
    >>> epo = mne.read_epochs(datasets.eeg_epochs(participant=1), preload=True)
    """
    return fetch(f"mne/epochs/participant{participant}-epo.fif")


def eeg_raw() -> str:
    """
    Fetch the raw EEG recording (sub-110, session 1, pre-condition).

    Returns
    -------
    str
        Local path to ``sub-110_session-1_pre.fif``.

    Example
    -------
    >>> import mne
    >>> from hypyp import datasets
    >>> raw = mne.io.read_raw_fif(datasets.eeg_raw(), preload=True)
    """
    return fetch("mne/raw/sub-110_session-1_pre.fif")


def eeglab_epochs() -> str:
    """
    Fetch the EEGLAB epoched, ICA-cleaned EEG dataset (.set file).

    The companion ``.fdt`` file is fetched automatically alongside the ``.set``.
    Both files are placed in the same cache directory so MNE can find them.

    Returns
    -------
    str
        Local path to ``eeglab_data_epochs_ica.set``.

    Example
    -------
    >>> import mne
    >>> from hypyp import datasets
    >>> epo = mne.io.read_epochs_eeglab(datasets.eeglab_epochs())
    """
    # Fetch the companion .fdt first (must be co-located with .set)
    fetch("eeglab/eeglab_data_epochs_ica.fdt")
    return fetch("eeglab/eeglab_data_epochs_ica.set")


def eeglab_raw() -> str:
    """
    Fetch the continuous EEGLAB EEG dataset (.set file).

    The companion ``.fdt`` file is fetched automatically.

    Returns
    -------
    str
        Local path to ``eeglab_data.set``.
    """
    fetch("eeglab/eeglab_data.fdt")
    return fetch("eeglab/eeglab_data.set")


def fnirs_samples(indices: list[int] | None = None) -> list[str]:
    """
    Fetch the fNIRS demo SNIRF files (mono-subject hyperscanning proxy).

    These are four 100-second windows of a single fNIRS recording, used
    to simulate two hyperscanning dyads in tutorials. They are **not**
    genuine hyperscanning data.

    Parameters
    ----------
    indices : list[int] | None
        Subset of sample indices to fetch (1–4). Default: all four [1, 2, 3, 4].

    Returns
    -------
    list[str]
        Local paths to the requested ``sample_{N}.snirf`` files, in order.

    Example
    -------
    >>> from hypyp import datasets
    >>> paths = datasets.fnirs_samples()
    >>> dyad_paths = {
    ...     'teamA': {'child': paths[0], 'parent': paths[1]},
    ...     'teamB': {'child': paths[2], 'parent': paths[3]},
    ... }
    """
    indices = indices or [1, 2, 3, 4]
    return [fetch(f"fnirs/samples/sample_{i}.snirf") for i in indices]


def fnirs_slow_breathing() -> str:
    """
    Fetch the fNIRS slow-breathing protocol recording.

    Returns
    -------
    str
        Local path to ``slow_breathing.snirf``.

    Example
    -------
    >>> import hypyp.fnirs as fnirs
    >>> from hypyp import datasets
    >>> rec = fnirs.Recording().load_file(datasets.fnirs_slow_breathing())
    """
    return fetch("fnirs/samples/slow_breathing.snirf")


def xdf_dyad_noise() -> str:
    """
    Fetch the XDF file with two EEG amplifiers recording pure noise.

    Two Starstim-32 headsets were connected without participants, producing
    only electronic noise. Useful for demonstrating XDF import and LSL
    stream selection.

    Returns
    -------
    str
        Local path to ``dyad-example-noise.xdf``.

    Example
    -------
    >>> from hypyp.xdf import XDFImport
    >>> from hypyp import datasets
    >>> xdf = XDFImport(datasets.xdf_dyad_noise(), convert_to_mne=True)
    """
    return fetch("xdf/dyad-example-noise.xdf")


def xdf_dyad_with_markers() -> str:
    """
    Fetch the XDF file with synthetic noise and event markers.

    Returns
    -------
    str
        Local path to ``dyad-example-with-markers.xdf``.

    Example
    -------
    >>> from hypyp.xdf import XDFImport
    >>> from hypyp import datasets
    >>> xdf = XDFImport(datasets.xdf_dyad_with_markers())
    >>> print(xdf.mne_raws[0].annotations)
    """
    return fetch("xdf/dyad-example-with-markers.xdf")


def xdf_clock_resets() -> str:
    """
    Fetch the XDF file containing LSL clock reset events.

    Returns
    -------
    str
        Local path to ``data_with_clock_resets.xdf``.
    """
    return fetch("xdf/data_with_clock_resets.xdf")


def fnirs_dcare(subject: int = 1) -> str:
    """
    Fetch a DCARE hyperscanning fNIRS recording (SNIRF format).

    DCARE_02 is a real hyperscanning dyad (two participants recorded simultaneously).

    Parameters
    ----------
    subject : int
        Subject index (1 or 2). Default: 1.

    Returns
    -------
    str
        Local path to ``DCARE_02_sub{N}.snirf``.

    Example
    -------
    >>> import hypyp.fnirs as fnirs
    >>> from hypyp import datasets
    >>> rec = fnirs.Recording().load_file(datasets.fnirs_dcare(subject=1))
    """
    return fetch(f"fnirs/DCARE/DCARE_02_sub{subject}.snirf")


def fnirs_fcs01_parent() -> str:
    """
    Fetch the FCS01 raw fNIRS recording for the parent participant (Homer2/NIRX format).

    All 12 companion files (``.hdr``, ``.wl1``, ``.wl2``, ``.dat``, ``.evt``,
    ``.inf``, ``.nirs``, ``.avg``, ``.set``, ``.tpl``, ``_config.txt``,
    ``_probeInfo.mat``) are fetched into the same cache directory so that MNE
    can locate them all relative to the ``.hdr`` entry point.

    Returns
    -------
    str
        Local path to ``NIRS-2019-09-28_002.hdr`` — pass this to
        ``fnirs.Recording().load_file()``.

    Example
    -------
    >>> import hypyp.fnirs as fnirs
    >>> from hypyp import datasets
    >>> rec = fnirs.Recording().load_file(datasets.fnirs_fcs01_parent())
    """
    _FCS01_PARENT_FILES = [
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.hdr",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.wl1",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.wl2",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.dat",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.evt",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.inf",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.nirs",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.avg",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.set",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002.tpl",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002_config.txt",
        "fnirs/FCS01/parent/NIRS-2019-09-28_002_probeInfo.mat",
    ]
    for f in _FCS01_PARENT_FILES:
        fetch(f)
    return fetch("fnirs/FCS01/parent/NIRS-2019-09-28_002.hdr")


def cache_dir() -> Path:
    """
    Return the local cache directory for HypypData files.

    Returns
    -------
    pathlib.Path
        Path to ``~/mne_data/HypypData/`` (or ``$MNE_DATA/HypypData/``).
    """
    return Path(_CACHE_DIR)
