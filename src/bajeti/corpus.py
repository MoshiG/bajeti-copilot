"""Corpus manifest and downloader.

Design choices:
- Standard library only (urllib), so `make fetch` works on a bare machine.
- Idempotent: files already present are verified, not re-downloaded.
- Every file is checked to be a real PDF and its SHA-256 is written to a lock file, so the
  exact corpus behind a set of evaluation results can be reproduced later.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

USER_AGENT = "BajetiCopilot/0.1 (research; +https://github.com/MoshiG/bajeti-copilot)"
DOC_TYPES = {"speech", "finance_act", "estimates"}
LANGUAGES = {"sw", "en"}

Fetcher = Callable[[str], bytes]


@dataclass
class Document:
    id: str
    title: str
    doc_type: str
    fiscal_year: str
    language: str
    url: str | None = None
    dspace_item: str | None = None
    file_match: str | None = None
    manual: bool = False
    page: str | None = None

    def __post_init__(self) -> None:
        if self.doc_type not in DOC_TYPES:
            raise ValueError(f"{self.id}: doc_type must be one of {sorted(DOC_TYPES)}")
        if self.language not in LANGUAGES:
            raise ValueError(f"{self.id}: language must be one of {sorted(LANGUAGES)}")
        sources = sum([bool(self.url), bool(self.dspace_item), self.manual])
        if sources != 1:
            raise ValueError(f"{self.id}: set exactly one of url, dspace_item or manual")
        if self.dspace_item and not self.file_match:
            raise ValueError(f"{self.id}: dspace_item needs file_match")

    @property
    def filename(self) -> str:
        return f"{self.id}.pdf"


@dataclass
class Manifest:
    documents: list[Document]
    repository_api: str = "https://repository.mof.go.tz/server/api"

    def get(self, doc_id: str) -> Document:
        for d in self.documents:
            if d.id == doc_id:
                return d
        raise KeyError(doc_id)


def load_manifest(path: str | Path) -> Manifest:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    docs = [Document(**d) for d in data.get("documents", [])]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document ids in manifest")
    api = data.get("repository_api", Manifest.repository_api)
    return Manifest(documents=docs, repository_api=api)


# --- HTTP --------------------------------------------------------------------------------------


def quote_url(url: str) -> str:
    """Ministry filenames contain spaces; percent-encode the path without double-encoding."""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/")
    return urllib.parse.urlunsplit(parts._replace(path=path))


def ssl_context() -> ssl.SSLContext | None:
    """Verify certificates with the operating system's trust store when possible.

    Some government servers send an incomplete certificate chain. Browsers and the OS verifier
    fetch the missing intermediate certificate; Python's default OpenSSL context does not, and
    fails with CERTIFICATE_VERIFY_FAILED. truststore hands verification to the OS (Keychain on
    macOS), so certificates are still verified. Verification is never switched off.
    """
    try:
        import truststore
    except ImportError:
        return None
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last: Exception | None = None
    context = ssl_context()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(quote_url(url), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout,  # noqa: S310 (fixed hosts)
                                        context=context) as resp:
                return resp.read()
        except Exception as exc:  # network errors vary by platform
            last = exc
            if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
                break                       # retrying will not fix a certificate problem
            time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def resolve_dspace(api: str, item_uuid: str, file_match: str, get: Fetcher) -> str:
    """Find the download URL of a file inside a DSpace 7 item via its REST API.

    items/{uuid}/bundles -> ORIGINAL bundle -> bitstreams -> file whose name contains file_match.
    """
    bundles = json.loads(get(f"{api}/core/items/{item_uuid}/bundles"))
    original = [
        b for b in bundles.get("_embedded", {}).get("bundles", []) if b.get("name") == "ORIGINAL"
    ]
    if not original:
        raise LookupError(f"item {item_uuid}: no ORIGINAL bundle")
    bitstreams_url = original[0]["_links"]["bitstreams"]["href"]
    bitstreams = json.loads(get(bitstreams_url)).get("_embedded", {}).get("bitstreams", [])
    needle = file_match.lower()
    for bs in bitstreams:
        name = bs.get("name", "")
        if needle in name.lower() and name.lower().endswith(".pdf"):
            return bs["_links"]["content"]["href"]
    names = [bs.get("name") for bs in bitstreams]
    raise LookupError(f"item {item_uuid}: no PDF matching {file_match!r} among {names}")


# --- Fetch -------------------------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


@dataclass
class FetchResult:
    id: str
    status: str            # downloaded | present | manual_missing | failed
    path: str | None = None
    sha256: str | None = None
    bytes: int = 0
    source: str | None = None
    message: str = ""


def fetch_document(doc: Document, raw_dir: Path, api: str, get: Fetcher = http_get) -> FetchResult:
    target = raw_dir / doc.filename
    if target.exists():
        data = target.read_bytes()
        if not is_pdf(data):
            return FetchResult(doc.id, "failed", str(target), message="existing file is not a PDF")
        return FetchResult(doc.id, "present", str(target), sha256(data), len(data))

    if doc.manual:
        return FetchResult(
            doc.id, "manual_missing", message=f"download from {doc.page} and save as {target}"
        )

    try:
        if doc.dspace_item:
            url = resolve_dspace(api, doc.dspace_item, doc.file_match or "", get)
        else:
            url = doc.url or ""
        data = get(url)
        if not is_pdf(data):
            raise ValueError("response is not a PDF (the site may have returned an HTML page)")
    except Exception as exc:
        hint = f" — try downloading manually from {doc.page}" if doc.page else ""
        return FetchResult(doc.id, "failed", message=f"{exc}{hint}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)  # atomic: a crash never leaves a half-written PDF behind
    return FetchResult(doc.id, "downloaded", str(target), sha256(data), len(data), source=url)


def fetch_all(
    manifest: Manifest,
    raw_dir: str | Path,
    only: set[str] | None = None,
    get: Fetcher = http_get,
) -> list[FetchResult]:
    raw = Path(raw_dir)
    results = [
        fetch_document(doc, raw, manifest.repository_api, get)
        for doc in manifest.documents
        if not only or doc.id in only
    ]
    write_lock(manifest, results, raw / "manifest.lock.json")
    return results


def write_lock(manifest: Manifest, results: list[FetchResult], path: Path) -> None:
    by_id = {d.id: d for d in manifest.documents}
    locked: dict[str, Any] = {}
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
    for r in results:
        if r.sha256:
            entry = {**asdict(by_id[r.id]), "sha256": r.sha256, "bytes": r.bytes}
            if r.source:
                entry["resolved_url"] = r.source
            locked[r.id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding="utf-8")
