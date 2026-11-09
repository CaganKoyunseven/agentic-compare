import re

from fastapi.testclient import TestClient

from agentic.server.app import STATIC_DIR, create_app
from scripts.make_mock_traces import write_all

VENDOR_DIR = STATIC_DIR / "vendor"
URL = re.compile(r"https?://", re.IGNORECASE)


def own_static_files():
    return [p for p in STATIC_DIR.rglob("*")
            if p.is_file() and VENDOR_DIR not in p.parents and p.name != ".gitkeep"]


def test_static_files_load_nothing_from_the_internet():
    files = own_static_files()
    assert files, "no static files found"
    offenders = [str(p.relative_to(STATIC_DIR)) for p in files
                 if URL.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_index_uses_vendored_d3():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'src="/static/vendor/d3.v7.min.js"' in html
    assert (VENDOR_DIR / "d3.v7.min.js").is_file()
    assert (VENDOR_DIR / "LICENSE-d3.txt").is_file()


def test_page_and_entry_script_are_served(tmp_path):
    write_all(tmp_path)
    client = TestClient(create_app(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<script type="module" src="/static/js/main.js">' in page.text
    for path in ("/static/js/main.js", "/static/css/demo.css", "/static/vendor/d3.v7.min.js"):
        assert client.get(path).status_code == 200, path


def test_every_module_import_exists():
    # Without a JS test framework, at least check that relative imports point to real files.
    for js in (STATIC_DIR / "js").glob("*.js"):
        for target in re.findall(r'from "(\./[^"]+)"', js.read_text(encoding="utf-8")):
            assert (js.parent / target).is_file(), f"{js.name} imports missing {target}"
