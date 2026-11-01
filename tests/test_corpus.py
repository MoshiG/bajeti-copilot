import json
from pathlib import Path

import pytest

from bajeti.corpus import Document, Manifest, fetch_all, load_manifest, quote_url, resolve_dspace

ROOT = Path(__file__).resolve().parents[1]
PDF = b"%PDF-1.7\nfake pdf body"
API = "https://repo.example/server/api"


def _doc(**kw):
    base = dict(id="d1", title="T", doc_type="speech", fiscal_year="2025/26", language="sw")
    return Document(**{**base, **kw})


def _dspace_routes(item="abc", names=("HOTUBA.pdf", "SPEECH OF BUDGET.pdf")):
    bundles = {"_embedded": {"bundles": [
        {"name": "LICENSE", "_links": {"bitstreams": {"href": f"{API}/lic"}}},
        {"name": "ORIGINAL", "_links": {"bitstreams": {"href": f"{API}/bs"}}},
    ]}}
    bitstreams = {"_embedded": {"bitstreams": [
        {"name": n, "_links": {"content": {"href": f"{API}/content/{i}"}}}
        for i, n in enumerate(names)
    ]}}
    return {
        f"{API}/core/items/{item}/bundles": json.dumps(bundles).encode(),
        f"{API}/bs": json.dumps(bitstreams).encode(),
        f"{API}/content/0": PDF + b" sw",
        f"{API}/content/1": PDF + b" en",
    }


def test_real_manifest_is_valid():
    manifest = load_manifest(ROOT / "data" / "corpus.yaml")
    assert len(manifest.documents) >= 10
    assert {d.doc_type for d in manifest.documents} == {"speech", "finance_act", "estimates"}
    assert {d.language for d in manifest.documents} == {"sw", "en"}


@pytest.mark.parametrize("kw, msg", [
    ({"url": "u", "manual": True}, "exactly one"),
    ({}, "exactly one"),
    ({"dspace_item": "x"}, "file_match"),
    ({"url": "u", "doc_type": "memo"}, "doc_type"),
    ({"url": "u", "language": "fr"}, "language"),
])
def test_document_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _doc(**kw)


def test_quote_url_encodes_spaces_once():
    url = "https://www.mof.go.tz/uploads/HOTUBA YA BAJETI 2021-22.pdf"
    assert quote_url(url) == "https://www.mof.go.tz/uploads/HOTUBA%20YA%20BAJETI%202021-22.pdf"
    assert quote_url(quote_url(url)) == quote_url(url)


def test_resolve_dspace_picks_matching_pdf():
    routes = _dspace_routes()
    assert resolve_dspace(API, "abc", "speech of", routes.__getitem__) == f"{API}/content/1"
    with pytest.raises(LookupError, match="no PDF matching"):
        resolve_dspace(API, "abc", "volume iv", routes.__getitem__)


def test_fetch_all_handles_every_source_kind(tmp_path):
    (tmp_path / "present.pdf").write_bytes(PDF)
    (tmp_path / "broken.pdf").write_bytes(b"<html>")
    routes = {"https://x/direct.pdf": PDF, "https://x/html": b"<html>error</html>",
              **_dspace_routes()}
    manifest = Manifest(repository_api=API, documents=[
        _doc(id="direct", url="https://x/direct.pdf"),
        _doc(id="repo", dspace_item="abc", file_match="SPEECH", page="https://x/item"),
        _doc(id="html", url="https://x/html", page="https://x/page"),
        _doc(id="manual", manual=True, page="https://x/collection"),
        _doc(id="present", url="https://x/unused"),
        _doc(id="broken", url="https://x/unused"),
    ])

    results = {r.id: r for r in fetch_all(manifest, tmp_path, get=routes.__getitem__)}

    assert results["direct"].status == "downloaded"
    assert results["repo"].status == "downloaded"
    assert (tmp_path / "repo.pdf").read_bytes().endswith(b" en")
    assert results["html"].status == "failed" and "https://x/page" in results["html"].message
    assert results["manual"].status == "manual_missing"
    assert results["present"].status == "present"
    assert results["broken"].status == "failed"
    assert not (tmp_path / "html.pdf").exists()

    lock = json.loads((tmp_path / "manifest.lock.json").read_text())
    assert set(lock) == {"direct", "repo", "present"}
    assert len(lock["direct"]["sha256"]) == 64
    assert lock["repo"]["resolved_url"] == f"{API}/content/1"


def test_fetch_only_limits_documents(tmp_path):
    manifest = Manifest(documents=[_doc(id="a", url="https://x/a"), _doc(id="b", url="https://x/b")])
    results = fetch_all(manifest, tmp_path, only={"a"}, get=lambda url: PDF)
    assert [r.id for r in results] == ["a"]
