"""Install pinned Gemma GGUF in the existing test container; no image changes."""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


def main():
    root = Path("/opt/vectorpro-models")
    if not root.is_dir():
        raise RuntimeError("existing model directory is missing")
    repo = "ggml-org/gemma-3-1b-it-GGUF"
    revision = "f9c28bcd85737ffc5aef028638d3341d49869c27"
    filename = "gemma-3-1b-it-Q8_0.gguf"
    with urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=60) as response:
        info = json.load(response)
    entry = next(x for x in info["siblings"] if x["rfilename"] == filename)
    expected = entry["lfs"]["sha256"]
    path = root / filename
    if not path.exists():
        temporary = path.with_suffix(".gguf.partial")
        print(f"downloading {filename} ({entry['size']} bytes)", flush=True)
        with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        with temporary.open("rb") as stream:
            downloaded = hashlib.file_digest(stream, "sha256").hexdigest()
        if downloaded != expected:
            raise RuntimeError("download checksum mismatch; partial file preserved")
        temporary.replace(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected or path.stat().st_size != entry["size"]:
        raise RuntimeError("installed Gemma does not match pinned source")
    manifest = {"repo": repo, "revision": revision, "quantization": "Q8_0", "filename": filename,
                "bytes": path.stat().st_size, "sha256": actual}
    (root / "gemma-download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
