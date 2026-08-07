"""Parser for #GQ directives in job scripts (sbatch-style).

Example script:

    #!/bin/bash
    #GQ gpus=2
    #GQ name=big-sweep
    python train.py --lr 1e-4

Directives are read from the top of the file; scanning stops at the first
line that is neither blank nor a comment, so directives can't appear after
the script body starts (same rule as sbatch).
"""

import re
from pathlib import Path
from typing import Dict

_DIRECTIVE_RE = re.compile(r"^#GQ\s+(.*)$")
_KV_RE = re.compile(r"([A-Za-z_-]+)=(\"[^\"]*\"|'[^']*'|\S+)")

KNOWN_KEYS = {"gpus", "name", "workdir", "vram"}


class JobScriptError(ValueError):
    pass


def parse_directives(text: str) -> Dict[str, str]:
    """Parse #GQ key=value directives from job-script text."""
    out: Dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break  # script body reached
        m = _DIRECTIVE_RE.match(stripped)
        if not m:
            continue  # ordinary comment / shebang
        body = m.group(1).strip()
        matched_len = 0
        for kv in _KV_RE.finditer(body):
            key, value = kv.group(1).lower(), kv.group(2)
            if value and value[0] in "\"'":
                value = value[1:-1]
            if key not in KNOWN_KEYS:
                raise JobScriptError(f"line {lineno}: unknown #GQ directive {key!r}")
            out[key] = value
            matched_len += len(kv.group(0))
        if not body or matched_len == 0:
            raise JobScriptError(f"line {lineno}: malformed #GQ directive: {line.strip()!r}")
    if "gpus" in out:
        try:
            n = int(out["gpus"])
        except ValueError:
            raise JobScriptError(f"gpus must be an integer, got {out['gpus']!r}")
        if n < 0:
            raise JobScriptError("gpus must be >= 0")
    return out


def parse_size(text: str) -> int:
    text = text.strip() 
    m = re.fullmatch(r"(\d+)(\w*)", text)
    if m==None:
        raise JobScriptError(f"invalid format: {text!r}, vram must be > 0 Examples: 12G, 12GB, 12GiB, 12g, 8192M or 8192MiB")
    size = m.group(1)   # '12'
    unit = m.group(2)  # 'GB'
    
    size=int(size)
    if size <=0:
        raise JobScriptError(f"invalid format: {text!r}, vram must be > 0 Examples: 12G, 12GB, 12GiB, 12g, 8192M or 8192MiB")

    if unit.lower()=='g' or unit.lower()=="gib" or unit.lower()=="gb":
        return size*1024

    if unit.lower()=='m' or unit.lower()=="mib" or unit.lower()=="mb" or unit.lower()=="":
        return size
    else:
        raise JobScriptError(f"invalid format: {text!r}, vram must be > 0 Examples: 12G, 12GB, 12GiB, 12g, 8192M or 8192MiB")



def parse_file(path: Path) -> Dict[str, str]:
    return parse_directives(path.read_text())
