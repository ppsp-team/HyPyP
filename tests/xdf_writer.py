"""Minimal XDF writer for synthetic test files.

Mirrors what pyxdf reads: a chunk is a variable-length size, a uint16 tag, a
uint32 stream id for stream chunks, then the content. Enough for a few signal
and marker streams with clock offsets; not a general-purpose writer.
"""

import struct

import numpy as np


def _varlen(n: int) -> bytes:
    if n < 256:
        return b"\x01" + struct.pack("<B", n)
    return b"\x04" + struct.pack("<I", n)


def _chunk(tag: int, content: bytes, stream_id: int | None = None) -> bytes:
    body = struct.pack("<H", tag)
    if stream_id is not None:
        body += struct.pack("<I", stream_id)
    body += content
    return _varlen(len(body)) + body


def _header_xml(stream: dict) -> bytes:
    fmt = stream.get("fmt", "float32")
    channels = ""
    if stream.get("labels"):
        channels = (
            "<channels>"
            + "".join(
                f"<channel><label>{label}</label><unit>microvolts</unit>"
                f"<type>{stream['type']}</type></channel>"
                for label in stream["labels"]
            )
            + "</channels>"
        )
    n_channels = len(stream["labels"]) if stream.get("labels") else stream.get("nch", 1)
    desc = f"<desc>{channels}</desc>" if channels else "<desc/>"
    return (
        '<?xml version="1.0"?><info>'
        f"<name>{stream['name']}</name><type>{stream['type']}</type>"
        f"<channel_count>{n_channels}</channel_count>"
        f"<nominal_srate>{stream['srate']}</nominal_srate>"
        f"<channel_format>{fmt}</channel_format><source_id>{stream['name']}</source_id>"
        f"{desc}</info>"
    ).encode()


def _samples(stream: dict) -> bytes:
    timestamps = np.asarray(stream["timestamps"], dtype=np.float64)
    out = _varlen(len(timestamps))
    if stream.get("fmt", "float32") == "string":
        for t, row in zip(timestamps, stream["values"]):
            out += b"\x08" + struct.pack("<d", t)
            for value in row:
                encoded = str(value).encode()
                out += _varlen(len(encoded)) + encoded
    else:
        values = np.asarray(stream["values"], dtype="<f4")
        for t, row in zip(timestamps, values):
            out += b"\x08" + struct.pack("<d", t) + row.tobytes()
    return out


def write_xdf(path, streams: list[dict]) -> str:
    """Write the streams to an XDF file and return its path.

    Each stream is a dict with ``id``, ``name``, ``type``, ``srate``,
    ``labels`` (or ``nch``), ``fmt`` (``"float32"`` or ``"string"``),
    ``timestamps``, ``values`` and optionally ``clock``, a list of
    ``(time, offset)`` pairs.
    """
    path = str(path)
    with open(path, "wb") as f:
        f.write(b"XDF:")
        f.write(_chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>'))
        for stream in streams:
            f.write(_chunk(2, _header_xml(stream), stream["id"]))
        for stream in streams:
            if len(stream["timestamps"]):
                f.write(_chunk(3, _samples(stream), stream["id"]))
            for t, offset in stream.get("clock", []):
                f.write(_chunk(4, struct.pack("<dd", t, offset), stream["id"]))
        for stream in streams:
            footer = b'<?xml version="1.0"?><info><first_timestamp>0</first_timestamp></info>'
            f.write(_chunk(6, footer, stream["id"]))
    return path
