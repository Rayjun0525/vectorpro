"""Replace the explicitly retired Gemma weights with ONE pinned Laya checkpoint.

Run only inside the existing vectorpro-test container. No Hub cache duplication.
"""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

REPO = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
SUBFOLDER = "multilingual"


def main():
    root = Path("/opt/vectorpro-models")
    if not root.is_dir():
        raise RuntimeError("existing model directory missing")
    with urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60) as response:
        info = json.load(response)
    entries = [e for e in info["siblings"] if e["rfilename"].startswith(SUBFOLDER + "/")]
    if not entries or not any(e["rfilename"].endswith("model.safetensors") for e in entries):
        raise RuntimeError("pinned checkpoint incomplete")
    retired = []
    # Exact user-authorized files only: keep MiniLM, programs and historical results.
    for name in ("gemma-3-1b-it-Q8_0.gguf", "gemma-download.json"):
        path = root / name
        if path.is_file():
            retired.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()
    target = root / "laya-multilingual"
    target.mkdir(exist_ok=True)
    previous = target / "download.json"
    if not retired and previous.exists():
        retired = json.loads(previous.read_text()).get("retired", [])
    manifest = {"repo": REPO, "revision": REVISION, "subfolder": SUBFOLDER,
                "sdk": "laya==0.3.26", "retired": retired, "files": []}
    for entry in entries:
        relative = entry["rfilename"].removeprefix(SUBFOLDER + "/")
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get("lfs", {}).get("sha256")
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".partial")
            print(f"downloading {relative}: {entry['size']} bytes", flush=True)
            with urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{entry['rfilename']}", timeout=180) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            if partial.stat().st_size != entry["size"]:
                raise RuntimeError("download size mismatch")
            with partial.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected and digest != expected:
                raise RuntimeError("download checksum mismatch")
            partial.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["size"] or (expected and digest != expected):
            raise RuntimeError("installed checkpoint mismatch")
        if not expected:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != entry["blobId"]:
                raise RuntimeError("configuration Git blob mismatch")
        manifest["files"].append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    manifest["installed_bytes"] = sum(e["bytes"] for e in manifest["files"])
    manifest["logical_bytes_saved"] = sum(e["bytes"] for e in retired) - manifest["installed_bytes"]
    (target / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
