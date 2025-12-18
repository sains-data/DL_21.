"""Utility to download model files safely.

Provides:
- download_url_to_path(url, dest_path, expected_sha256=None)
- download_from_github_release(repo, asset_name=None, token=None, dest_path=None)
- ensure_model(url=None, dest_path=None, expected_sha256=None, force=False, github_repo=None, asset_name=None, github_token=None)

This is intentionally lightweight and uses the GitHub Releases API when asked to fetch the latest asset.
"""
import hashlib
import os
import shutil
import tempfile
from typing import Optional

try:
    import requests
except Exception as e:
    requests = None


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url_to_path(url: str, dest_path: str, expected_sha256: Optional[str] = None, chunk_size: int = 8192):
    if requests is None:
        raise RuntimeError("The 'requests' package is required to download files. Install with 'pip install requests'.")

    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp")
    os.close(tmp_fd)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)

        if expected_sha256:
            actual = sha256_of_file(tmp_path)
            if actual.lower() != expected_sha256.lower():
                raise RuntimeError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")

        shutil.move(tmp_path, dest_path)
        return dest_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def download_from_github_release(repo: str, asset_name: Optional[str] = None, token: Optional[str] = None, dest_path: Optional[str] = None):
    """Download an asset from the latest GitHub release for `repo` (owner/repo).

    If `asset_name` is None, picks the first asset whose name contains 'best' or endswith '.h5'.
    Returns the path to the downloaded file.
    """
    if requests is None:
        raise RuntimeError("The 'requests' package is required to download files. Install with 'pip install requests'.")

    owner_repo = repo.strip()
    api_url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    r = requests.get(api_url, headers=headers, timeout=20)
    r.raise_for_status()
    release = r.json()
    assets = release.get('assets', [])
    if not assets:
        raise RuntimeError(f"No assets found in latest release of {repo}")

    chosen = None
    if asset_name:
        for a in assets:
            if a.get('name') == asset_name:
                chosen = a
                break

    if chosen is None:
        # fallback heuristics
        for a in assets:
            name = a.get('name', '').lower()
            if asset_name and asset_name.lower() in name:
                chosen = a
                break
        if chosen is None:
            for a in assets:
                name = a.get('name', '').lower()
                if 'best' in name or name.endswith('.h5') or name.endswith('.keras'):
                    chosen = a
                    break

    if chosen is None:
        # last resort: pick first asset
        chosen = assets[0]

    download_url = chosen.get('browser_download_url')
    filename = chosen.get('name')
    if dest_path is None:
        dest_path = os.path.join('models', filename)

    return download_url_to_path(download_url, dest_path)


def ensure_model(url: Optional[str] = None, dest_path: Optional[str] = None, expected_sha256: Optional[str] = None, force: bool = False, github_repo: Optional[str] = None, asset_name: Optional[str] = None, github_token: Optional[str] = None):
    """High-level helper. If `github_repo` is provided, attempts to download from its latest release.

    Otherwise, if `url` provided, downloads from that URL. If the file already exists and `force` is False,
    the existing file is returned.
    """
    if dest_path is None:
        raise ValueError("dest_path must be provided")

    if os.path.exists(dest_path) and not force:
        # optional checksum verify
        if expected_sha256:
            try:
                actual = sha256_of_file(dest_path)
                if actual.lower() == expected_sha256.lower():
                    return dest_path
                else:
                    # mismatch → redownload
                    pass
            except Exception:
                pass
        else:
            return dest_path

    # ensure parent exists
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)

    if github_repo:
        return download_from_github_release(github_repo, asset_name=asset_name, token=github_token, dest_path=dest_path)

    if url:
        return download_url_to_path(url, dest_path, expected_sha256=expected_sha256)

    raise ValueError("Either url or github_repo must be provided to download a model.")
