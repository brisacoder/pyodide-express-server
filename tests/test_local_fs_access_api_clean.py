"""Clean replacement for corrupted test_local_fs_access_api module.

This file contains the authoritative local filesystem access tests.
Once stability is confirmed, the legacy file can be removed entirely.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

try:
    from .conftest import execute_python_code, validate_api_contract
except ImportError:  # pragma: no cover
    from conftest import execute_python_code, validate_api_contract  # type: ignore


@pytest.fixture(scope="class")
def test_data_paths() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    host_plots_dir = project_root / "pyodide_data" / "plots"
    host_uploads_dir = project_root / "pyodide_data" / "uploads"
    host_plots_dir.mkdir(parents=True, exist_ok=True)
    host_uploads_dir.mkdir(parents=True, exist_ok=True)
    return {
        "session_id": uuid.uuid4().hex[:8],
        "host_plots_dir": host_plots_dir,
        "host_uploads_dir": host_uploads_dir,
    }


@pytest.fixture(scope="class")
def fs_test_files(test_data_paths: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    sid = test_data_paths["session_id"]
    plots = test_data_paths["host_plots_dir"]
    uploads = test_data_paths["host_uploads_dir"]
    text_content = "Hello from local FS via API!"
    json_obj = {"message": "Test JSON data", "number": 42, "array": [1, 2, 3]}
    json_text = json.dumps(json_obj)
    binary_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

    files = {
        "text": plots / f"test_fs_access_{sid}.txt",
        "json": uploads / f"test_data_{sid}.json",
        "binary": plots / f"test_binary_{sid}.dat",
        "multi1": plots / f"multi_test_1_{sid}.txt",
        "multi2": plots / f"multi_test_2_{sid}.txt",
    }

    files["text"].write_text(text_content, encoding="utf-8")
    files["json"].write_text(json_text, encoding="utf-8")
    files["binary"].write_bytes(binary_bytes)
    files["multi1"].write_text("Content of file 1", encoding="utf-8")
    files["multi2"].write_text("Content of file 2", encoding="utf-8")

    for p in files.values():
        assert p.exists()

    yield {
        "session_id": sid,
        "text_content": text_content,
        "json_text": json_text,
        "file_count": 5,
    }

    for p in files.values():  # pragma: no cover
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


class TestLocalFSAccessClean:
    @pytest.mark.api
    @pytest.mark.filesystem
    def test_basic_file_reading(self, _server_ready: None, fs_test_files: Dict[str, Any]) -> None:  # noqa: ARG002
        sid = fs_test_files["session_id"]
        expected = fs_test_files["text_content"]
        code = f"""
from pathlib import Path
p = Path('/home/pyodide/plots/test_fs_access_{sid}.txt')
print('Exists:', p.exists())
print('Content:', p.read_text(encoding='utf-8'))
"""
        r = execute_python_code(code)
        assert r["success"] is True
        out = r["data"]["stdout"]
        assert "Exists: True" in out
        assert expected in out

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_json_parsing(self, _server_ready: None, fs_test_files: Dict[str, Any]) -> None:  # noqa: ARG002
        sid = fs_test_files["session_id"]
        code = f"""
import json
from pathlib import Path
p = Path('/home/pyodide/uploads/test_data_{sid}.json')
obj = json.loads(p.read_text(encoding='utf-8'))
print('Keys:', sorted(obj.keys()))
print('Number:', obj['number'])
"""
        r = execute_python_code(code)
        out = r["data"]["stdout"]
        assert "Keys:" in out and "'array'" in out
        assert "Number: 42" in out

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_binary_header(self, _server_ready: None, fs_test_files: Dict[str, Any]) -> None:  # noqa: ARG002
        sid = fs_test_files["session_id"]
        code = f"""
from pathlib import Path
p = Path('/home/pyodide/plots/test_binary_{sid}.dat')
print('Header OK' if p.read_bytes().startswith(b'\x89PNG') else 'Header BAD')
"""
        r = execute_python_code(code)
        assert "Header OK" in r["data"]["stdout"]

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_batch_files(self, _server_ready: None, fs_test_files: Dict[str, Any]) -> None:  # noqa: ARG002
        sid = fs_test_files["session_id"]
        code = f"""
from pathlib import Path
plots = Path('/home/pyodide/plots')
files = [
  'test_fs_access_{sid}.txt',
  'multi_test_1_{sid}.txt',
  'multi_test_2_{sid}.txt'
]
processed = 0
for name in files:
    p = plots / name
    if p.exists():
        processed += 1
print('Processed', processed)
"""
        r = execute_python_code(code)
        assert "Processed 3" in r["data"]["stdout"]

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_error_missing(self, _server_ready: None) -> None:  # noqa: ARG002
        missing = f"nofile_{uuid.uuid4().hex[:6]}.txt"
        code = f"""
from pathlib import Path
p = Path('/home/pyodide/plots/{missing}')
print('Exists:', p.exists())
try:
    p.read_text(encoding='utf-8')
except FileNotFoundError:
    print('Caught FileNotFoundError')
"""
        r = execute_python_code(code)
        out = r["data"]["stdout"]
        assert "Exists: False" in out and "Caught FileNotFoundError" in out

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_complex_workflow(self, _server_ready: None, fs_test_files: Dict[str, Any]) -> None:  # noqa: ARG002
        sid = fs_test_files["session_id"]
        code = f"""
import json
from pathlib import Path
plots = Path('/home/pyodide/plots')
uploads = Path('/home/pyodide/uploads')
files = [
  plots / 'test_fs_access_{sid}.txt',
  uploads / 'test_data_{sid}.json',
  plots / 'test_binary_{sid}.dat',
  plots / 'multi_test_1_{sid}.txt',
  plots / 'multi_test_2_{sid}.txt'
]
count = 0
for p in files:
    if not p.exists():
        raise RuntimeError('missing ' + p.name)
    count += 1
print('Workflow processed', count)
"""
        r = execute_python_code(code)
        assert "Workflow processed 5" in r["data"]["stdout"]

    @pytest.mark.api
    @pytest.mark.filesystem
    def test_api_contract(self, _server_ready: None) -> None:  # noqa: ARG002
        code = """
from pathlib import Path
print(Path('/home/pyodide/plots').exists())
"""
        r = execute_python_code(code)
        validate_api_contract(r)
        assert r["success"] is True
        assert r["error"] is None
        print('API contract verified for clean FS tests')
