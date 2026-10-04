"""OS-independent byte/handle interface; native I/O lives at this boundary.

These primitives are provided execution machinery, not learned capabilities.
Contexts are transient and are never persisted in vector program files.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import torch

from vectorpro.bits import BitCodec
from vectorpro.execution import BitExecutable

_CURRENT: ContextVar[HostContext | None] = ContextVar("vectorpro_host", default=None)
ARITIES = {"buffer.new": 1, "buffer.length": 1, "buffer.get": 2,
           "buffer.set": 3, "file.read": 1, "file.write": 2}
BASE_OPERATIONS = tuple(ARITIES)
ARITIES.update({"path.exists": 1, "file.remove": 1, "file.move": 2,
                "directory.create": 1, "directory.remove": 1, "directory.list": 1})
HOST_TYPES = {"buffer.new": (("value",), "buffer"),
              "buffer.length": (("buffer",), "value"),
              "buffer.get": (("buffer", "value"), "value"),
              "buffer.set": (("buffer", "value", "value"), "value"),
              "file.read": (("path",), "buffer"),
              "file.write": (("path", "buffer"), "value"),
              "path.exists": (("path",), "value"),
              "file.remove": (("path",), "value"),
              "file.move": (("path", "path"), "value"),
              "directory.create": (("path",), "value"),
              "directory.remove": (("path",), "value"),
              "directory.list": (("path",), "buffer")}


@dataclass
class HostContext:
    root: Path
    max_buffer_bytes: int = 16 * 1024 * 1024
    buffers: dict[int, bytearray] = field(default_factory=dict, init=False)
    events: list[dict] = field(default_factory=list, init=False)
    _next_handle: int = field(default=1, init=False)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        if self.max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")

    def put(self, data: bytes | bytearray) -> int:
        if len(data) > self.max_buffer_bytes:
            raise ValueError("buffer exceeds configured byte limit")
        handle = self._next_handle
        self._next_handle += 1
        self.buffers[handle] = bytearray(data)
        return handle

    def _path(self, handle: int) -> Path:
        try:
            relative = Path(MemoryHostContext.normalize(bytes(self.buffers[handle]).decode("utf-8")))
        except ValueError as error:
            raise ValueError("file path escapes the host root or is not a portable relative path") from error
        if relative.is_absolute():
            raise ValueError("file paths must be relative to the host root")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("file path escapes the host root")
        return path

    def invoke(self, operation: str, args: tuple[int, ...], width: int) -> int:
        if operation in ("buffer.new", "file.read", "directory.list") and self._next_handle >= 1 << width:
            raise ValueError("register width cannot represent another buffer handle")
        if operation == "buffer.new":
            if args[0] > self.max_buffer_bytes:
                raise ValueError("buffer exceeds configured byte limit")
            result = self.put(bytearray(args[0]))
        elif operation == "buffer.length":
            result = len(self.buffers[args[0]])
        elif operation == "buffer.get":
            result = self.buffers[args[0]][args[1]]
        elif operation == "buffer.set":
            self.buffers[args[0]][args[1]] = args[2]
            result = args[2]
        elif operation == "file.read":
            path = self._path(args[0])
            data = self._read_file(path)
            result = self.put(data)
        elif operation == "file.write":
            path = self._path(args[0])
            data = self.buffers[args[1]]
            if len(data) >= 1 << width:
                raise ValueError("register width cannot represent the write count")
            result = self._write_file(path, data)
        elif operation == "path.exists":
            result = int(self._exists(self._path(args[0])))
        elif operation == "file.remove":
            result = self._remove_file(self._path(args[0]))
        elif operation == "file.move":
            result = self._move_file(self._path(args[0]), self._path(args[1]))
        elif operation == "directory.create":
            result = self._create_directory(self._path(args[0]))
        elif operation == "directory.remove":
            result = self._remove_directory(self._path(args[0]))
        elif operation == "directory.list":
            data = self._list_directory(self._path(args[0]))
            result = self.put(data)
        else:
            raise ValueError(f"unknown host operation {operation!r}")
        if not 0 <= result < 1 << width:
            raise ValueError("register width cannot represent the host result")
        self.events.append({"operation": operation, "arguments": list(args), "result": result})
        return result

    def _read_file(self, path: Path) -> bytes:
        with path.open("rb") as stream:
            return stream.read(self.max_buffer_bytes + 1)

    def _write_file(self, path: Path, data: bytearray) -> int:
        return path.write_bytes(data)

    def _exists(self, path: Path) -> bool:
        return path.exists()

    def _remove_file(self, path: Path) -> int:
        if path.is_dir():
            raise IsADirectoryError(str(path))
        path.unlink()
        return 1

    def _move_file(self, source: Path, destination: Path) -> int:
        if source.is_dir() or destination.is_dir():
            raise IsADirectoryError(str(source))
        source.replace(destination)
        return 1

    def _create_directory(self, path: Path) -> int:
        path.mkdir()
        return 1

    def _remove_directory(self, path: Path) -> int:
        path.rmdir()
        return 1

    def _list_directory(self, path: Path) -> bytes:
        # Deterministic portable names; no shell, metadata formatting or task recipe.
        result = bytearray()
        for name in sorted(p.name for p in path.iterdir()):
            result.extend(name.encode("utf-8") + b"\0")
            if len(result) > self.max_buffer_bytes:
                raise ValueError("directory listing exceeds configured byte limit")
        return bytes(result)

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)


class MemoryHostContext(HostContext):
    """Fresh, isolated file state for synthesis; never invokes native file I/O."""
    def __init__(self, files: dict[str, bytes], max_buffer_bytes: int = 16 * 1024 * 1024,
                 directories=None):
        super().__init__(Path("/"), max_buffer_bytes)
        self.files = {self.normalize(name): bytes(data) for name, data in files.items()}
        self.directories = self.directory_state(self.files, directories)

    @classmethod
    def directory_state(cls, files, directories=None):
        if directories is not None and (not isinstance(directories, (list, tuple, set))
                or any(not isinstance(p, str) for p in directories)):
            raise ValueError("directory snapshots must be collections of paths")
        result = {cls.normalize(p) for p in (directories or ())}
        for name in [*files, *result]:
            result.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
        if set(files) & result:
            raise ValueError("a snapshot path cannot be both file and directory")
        return result

    @staticmethod
    def normalize(name: str) -> str:
        path = PurePosixPath(name)
        if not name or "\0" in name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("virtual paths must be portable relative paths")
        if str(path) == ".":
            raise ValueError("virtual path must name a file")
        return str(path)

    def _path(self, handle: int) -> str:
        return self.normalize(bytes(self.buffers[handle]).decode("utf-8"))

    def _read_file(self, path: str) -> bytes:
        if path in self.directories:
            raise IsADirectoryError(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def _write_file(self, path: str, data: bytearray) -> int:
        self._require_parent(path)
        if path in self.directories:
            raise IsADirectoryError(path)
        self.files[path] = bytes(data)
        return len(data)

    def _require_parent(self, path):
        parent = str(PurePosixPath(path).parent)
        if parent != "." and parent not in self.directories:
            raise FileNotFoundError(parent)

    def _exists(self, path):
        return path in self.files or path in self.directories

    def _remove_file(self, path):
        if path in self.directories:
            raise IsADirectoryError(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]
        return 1

    def _move_file(self, source, destination):
        if source in self.directories or destination in self.directories:
            raise IsADirectoryError(source)
        if source not in self.files:
            raise FileNotFoundError(source)
        self._require_parent(destination)
        data = self.files.pop(source)
        self.files[destination] = data
        return 1

    def _create_directory(self, path):
        self._require_parent(path)
        if self._exists(path):
            raise FileExistsError(path)
        self.directories.add(path)
        return 1

    def _remove_directory(self, path):
        if path not in self.directories:
            if path in self.files:
                raise NotADirectoryError(path)
            raise FileNotFoundError(path)
        if any(p.startswith(path + "/") for p in [*self.files, *self.directories]):
            raise OSError("directory is not empty")
        self.directories.remove(path)
        return 1

    def _list_directory(self, path):
        if path not in self.directories:
            if path in self.files:
                raise NotADirectoryError(path)
            raise FileNotFoundError(path)
        names = sorted(PurePosixPath(p).name for p in [*self.files, *self.directories]
                       if str(PurePosixPath(p).parent) == path)
        data = b"".join(n.encode("utf-8") + b"\0" for n in names)
        if len(data) > self.max_buffer_bytes:
            raise ValueError("directory listing exceeds configured byte limit")
        return data


class HostExecutable(BitExecutable):
    def __init__(self, operation: str):
        if operation not in ARITIES:
            raise ValueError(f"unknown host operation {operation!r}")
        self.operation = operation
        self.arity = ARITIES[operation]

    @property
    def effects(self) -> bool:
        return True

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer) -> torch.Tensor:
        context = _CURRENT.get()
        if context is None:
            raise RuntimeError("host operations require an active HostContext")
        if len(operands) != 1:
            raise ValueError("effectful calls require a single execution lane")
        args = tuple(BitCodec.decode(operands[0]))
        width = operands.shape[-1]
        return BitCodec.encode([context.invoke(self.operation, args, width)], width)

    def compiled(self):
        return self
