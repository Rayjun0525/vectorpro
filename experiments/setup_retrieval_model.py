"""Download one pinned official embedding model into the existing container only."""
import json
from pathlib import Path
from urllib.request import urlopen
import shutil

repo = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
root = Path("/opt/vectorpro-models/multilingual-minilm")
root.mkdir(parents=True, exist_ok=True)
manifest = root / "download.json"
if manifest.exists():
    revision = json.loads(manifest.read_text())["revision"]
else:
    with urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as stream:
        revision = json.load(stream)["sha"]
    manifest.write_text(json.dumps({"repo": repo, "revision": revision}))
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"):
    path = root / name
    if path.exists():
        continue
    print(f"downloading {name} at {revision}", flush=True)
    temporary = root / (name + ".partial")
    with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(path)
print(json.dumps({"repo": repo, "revision": revision, "bytes": sum(p.stat().st_size for p in root.iterdir())}), flush=True)
