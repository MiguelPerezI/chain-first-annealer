"""Record current export provenance without assigning it to historical runs."""
import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def environment_record(init_path=None):
    directory = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(directory), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = ['gpu_sc.py', 'validation.py', 'provenance.py', 'independent_check.py']
    versions = {}
    for package in ['numpy', 'numba', 'llvmlite', 'scipy']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = git('status', '--porcelain')
    result = dict(exported_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), platform=platform.platform(),
                  packages=versions, git_commit=git('rev-parse', 'HEAD'),
                  git_dirty=None if status is None else bool(status),
                  source_sha256={f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                                 for f in files},
                  scope='Current export environment; does not attest historical search hardware or timing.')
    if init_path:
        result['continuation_input'] = dict(path=str(init_path),
            sha256=hashlib.sha256(Path(init_path).read_bytes()).hexdigest())
    return result
