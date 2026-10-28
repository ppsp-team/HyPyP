"""Minimal XDF writer mirroring pyxdf's reader (chunk = varlen length, uint16 tag, [uint32 stream id], content)."""

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


def _header_xml(s: dict) -> bytes:
    fmt = s.get("fmt", "float32")
    chans = ""
    if s.get("labels"):
        chans = "<channels>" + "".join(
            f"<channel><label>{l}</label><unit>microvolts</unit><type>{s['type']}</type></channel>"
            for l in s["labels"]
        ) + "</channels>"
    nch = len(s["labels"]) if s.get("labels") else s.get("nch", 1)
    desc = f"<desc>{chans}</desc>" if chans else "<desc/>"
    return (
        '<?xml version="1.0"?><info>'
        f"<name>{s['name']}</name><type>{s['type']}</type>"
        f"<channel_count>{nch}</channel_count><nominal_srate>{s['srate']}</nominal_srate>"
        f"<channel_format>{fmt}</channel_format><source_id>{s['name']}</source_id>"
        f"{desc}</info>"
    ).encode()


def _samples(s: dict) -> bytes:
    ts = np.asarray(s["timestamps"], dtype=np.float64)
    out = _varlen(len(ts))
    if s.get("fmt", "float32") == "string":
        for t, row in zip(ts, s["values"]):
            out += b"\x08" + struct.pack("<d", t)
            for v in row:
                b = str(v).encode()
                out += _varlen(len(b)) + b
    else:
        vals = np.asarray(s["values"], dtype="<f4")
        for t, row in zip(ts, vals):
            out += b"\x08" + struct.pack("<d", t) + row.tobytes()
    return out


def write_xdf(path: str, streams: list[dict]) -> str:
    """Each stream dict: id, name, type, srate, labels|nch, fmt, timestamps, values, clock=[(time, offset),...]."""
    with open(path, "wb") as f:
        f.write(b"XDF:")
        f.write(_chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>'))
        for s in streams:
            f.write(_chunk(2, _header_xml(s), s["id"]))
        for s in streams:
            if len(s["timestamps"]):
                f.write(_chunk(3, _samples(s), s["id"]))
            for t, off in s.get("clock", []):
                f.write(_chunk(4, struct.pack("<dd", t, off), s["id"]))
        for s in streams:
            f.write(_chunk(6, b'<?xml version="1.0"?><info><first_timestamp>0</first_timestamp></info>', s["id"]))
    return path
