"""Small reproducibility and atomic-file helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import select
import shutil
import signal
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO


MAX_JSON_ARTIFACT_BYTES = 16 << 20
MAX_JSON_NESTING_DEPTH = 128
MAX_GIT_COMMAND_OUTPUT_BYTES = 1 << 20
MAX_GIT_WORKTREE_BYTES = 1 << 30
_JSON_READ_CHUNK_BYTES = 65536
_COMMAND_READ_CHUNK_BYTES = 65536
_COMMAND_TIMEOUT_SECONDS = 5.0


def stable_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_digest(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()


def exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON-compatible values without Python bool/int aliasing."""
    try:
        return stable_json(left) == stable_json(right)
    except (TypeError, ValueError):
        return False


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digest_nofollow(path: Path, *, max_bytes: int | None = None) -> str:
    """Hash one bounded stable regular file without following path symlinks."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("max_bytes must be null or a nonnegative integer")
    boundary, parts = _relative_parts(path, path.parent)
    parent_descriptor = _open_directory_chain(boundary, parts[:-1], create=False)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{path} must be a regular file")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise ValueError(f"{path} exceeds the {max_bytes}-byte size limit")
        digest = hashlib.sha256()
        total_bytes = 0
        while block := os.read(descriptor, 1024 * 1024):
            total_bytes += len(block)
            if max_bytes is not None and total_bytes > max_bytes:
                raise ValueError(f"{path} exceeds the {max_bytes}-byte size limit")
            digest.update(block)
        finished = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(finished):
            raise ValueError(f"{path} changed while being hashed")
        return digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def read_bytes_nofollow(
    path: Path,
    *,
    root: Path | None = None,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
) -> bytes:
    """Read one bounded stable regular file without following path symlinks."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    boundary, parts = _relative_parts(path, root or path.parent)
    parent_descriptor = _open_directory_chain(boundary, parts[:-1], create=False)
    descriptor: int | None = None
    try:
        entry = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError(f"{path} must be a regular file, not a symlink or special file")
        if entry.st_size > max_bytes:
            raise ValueError(f"{path} exceeds the {max_bytes}-byte size limit")
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _file_signature(entry) != _file_signature(opened)
        ):
            raise ValueError(f"{path} changed before it was opened")
        chunks: list[bytes] = []
        bytes_read = 0
        while block := os.read(descriptor, min(65536, max_bytes + 1 - bytes_read)):
            chunks.append(block)
            bytes_read += len(block)
            if bytes_read > max_bytes:
                raise ValueError(f"{path} exceeds the {max_bytes}-byte size limit")
        finished = os.fstat(descriptor)
        if (
            bytes_read != entry.st_size
            or _file_signature(opened) != _file_signature(finished)
        ):
            raise ValueError(f"{path} changed while being read")
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_parts(path: Path, root: Path) -> tuple[Path, tuple[str, ...]]:
    absolute_root = _absolute_lexical(root)
    absolute_path = _absolute_lexical(path)
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise ValueError(f"{path} is outside boundary {root}") from exc
    if not relative.parts:
        raise ValueError(f"{path} must name an entry beneath boundary {root}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{path} contains an unsafe path component")
    return absolute_root, relative.parts


def _open_directory_chain(
    root: Path, parts: tuple[str, ...], *, create: bool
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    absolute_root = _absolute_lexical(root)
    descriptor = os.open(absolute_root.anchor, flags)
    try:
        for part in absolute_root.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{root} must be a directory")
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_json_nesting(payload: bytes, path: Path, max_depth: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for value in payload:
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value in (ord("["), ord("{")):
            depth += 1
            if depth > max_depth:
                raise ValueError(
                    f"{path} exceeds the JSON nesting limit {max_depth}"
                )
        elif value in (ord("]"), ord("}")):
            depth = max(0, depth - 1)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON token is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key is forbidden in JSON objects: {key!r}")
        result[key] = value
    return result


def _contains_utf8_replacement(value: Any) -> bool:
    if type(value) is str:
        return "\ufffd" in value
    if type(value) is list:
        return any(_contains_utf8_replacement(item) for item in value)
    if type(value) is dict:
        return any(
            _contains_utf8_replacement(key) or _contains_utf8_replacement(item)
            for key, item in value.items()
        )
    return False


def parse_bounded_json_bytes(
    payload: bytes,
    *,
    source: str,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
    max_depth: int = MAX_JSON_NESTING_DEPTH,
) -> Any:
    """Parse bounded strict JSON bytes with duplicate and nonfinite rejection."""
    if len(payload) > max_bytes:
        raise ValueError(f"{source} exceeds the {max_bytes}-byte JSON size limit")
    _validate_json_nesting(payload, Path(source), max_depth)
    try:
        parsed = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError(f"{source} does not contain valid bounded JSON: {exc}") from exc
    if _contains_utf8_replacement(parsed):
        raise ValueError(f"{source} contains invalid UTF-8 replacement evidence")
    return parsed


def read_json_at_nofollow_with_size(
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    expected_metadata: os.stat_result | None = None,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
    max_depth: int = MAX_JSON_NESTING_DEPTH,
) -> tuple[Any, int]:
    """Read stable JSON relative to an already pinned parent directory."""
    if not name or name in {".", ".."} or os.sep in name:
        raise ValueError(f"{path} has an unsafe descriptor-relative entry name")
    entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if expected_metadata is not None and _file_signature(entry) != _file_signature(
        expected_metadata
    ):
        raise ValueError(f"{path} changed before it could be read")
    if not stat.S_ISREG(entry.st_mode):
        raise ValueError(
            f"{path} must be a regular file, not a symlink or special file"
        )
    if entry.st_size > max_bytes:
        raise ValueError(f"{path} exceeds the {max_bytes}-byte JSON size limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{path} must be a regular file")
        if _file_signature(entry) != _file_signature(opened):
            raise ValueError(f"{path} changed before it could be read")
        if opened.st_size > max_bytes:
            raise ValueError(f"{path} exceeds the {max_bytes}-byte JSON size limit")

        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(
                descriptor,
                min(_JSON_READ_CHUNK_BYTES, max_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > max_bytes:
            raise ValueError(f"{path} exceeds the {max_bytes}-byte JSON size limit")
        finished = os.fstat(descriptor)
        if _file_signature(opened) != _file_signature(finished):
            raise ValueError(f"{path} changed while being read")
        if len(payload) != finished.st_size:
            raise ValueError(f"{path} changed while being read")
    finally:
        os.close(descriptor)

    encoded = bytes(payload)
    parsed = parse_bounded_json_bytes(
        encoded,
        source=str(path),
        max_bytes=max_bytes,
        max_depth=max_depth,
    )
    return parsed, len(encoded)


def read_json_nofollow_with_size(
    path: Path,
    *,
    root: Path | None = None,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
    max_depth: int = MAX_JSON_NESTING_DEPTH,
) -> tuple[Any, int]:
    """Read stable regular JSON and return its verified encoded byte count."""
    boundary, parts = _relative_parts(path, root or path.parent)
    parent_descriptor = _open_directory_chain(
        boundary, parts[:-1], create=False
    )
    try:
        return read_json_at_nofollow_with_size(
            parent_descriptor,
            parts[-1],
            path,
            max_bytes=max_bytes,
            max_depth=max_depth,
        )
    finally:
        os.close(parent_descriptor)


def read_json_nofollow(
    path: Path,
    *,
    root: Path | None = None,
    max_bytes: int = MAX_JSON_ARTIFACT_BYTES,
    max_depth: int = MAX_JSON_NESTING_DEPTH,
) -> Any:
    """Read one stable regular JSON file without following path symlinks."""
    parsed, _ = read_json_nofollow_with_size(
        path,
        root=root,
        max_bytes=max_bytes,
        max_depth=max_depth,
    )
    return parsed


def ensure_directory_nofollow(path: Path, *, root: Path) -> None:
    boundary, parts = _relative_parts(path / ".entry", root)
    descriptor = _open_directory_chain(boundary, parts[:-1], create=True)
    os.close(descriptor)


def ensure_absolute_directory_nofollow(path: Path) -> None:
    """Create an absolute directory without following any path-component symlink."""
    descriptor = open_absolute_directory_nofollow(path, create=True)
    os.close(descriptor)


def open_absolute_directory_nofollow(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory chain without following symlinks; caller closes."""
    absolute = _absolute_lexical(path)
    return _open_directory_chain(
        Path(absolute.anchor),
        absolute.parts[1:],
        create=create,
    )


@contextmanager
def exclusive_directory_lock(
    path: Path,
    *,
    blocking: bool = False,
) -> Iterator[int]:
    """Hold an exclusive advisory lock on one no-follow directory descriptor."""
    absolute = _absolute_lexical(path)
    descriptor = _open_directory_chain(
        Path(absolute.anchor),
        absolute.parts[1:],
        create=False,
    )
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    locked = False
    try:
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise ValueError(f"directory has an active writer: {path}") from exc
        locked = True
        yield descriptor
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _existing_output_is_valid(parent_descriptor: int, name: str, path: Path) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            f"{path} must be absent or a regular file, not a symlink or special file"
        )


@contextmanager
def _atomic_writer(
    path: Path,
    *,
    root: Path | None,
    binary: bool,
    newline: str | None = None,
) -> Iterator[Any]:
    boundary, parts = _relative_parts(path, root or path.parent)
    parent_descriptor = _open_directory_chain(boundary, parts[:-1], create=True)
    temporary_name: str | None = None
    descriptor: int | None = None
    published = False
    try:
        temporary_name = f".{parts[-1]}.{secrets.token_hex(16)}.tmp"
        _existing_output_is_valid(parent_descriptor, parts[-1], path)
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o644,
            dir_fd=parent_descriptor,
        )
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": newline}
        with os.fdopen(descriptor, mode, **kwargs) as stream:
            descriptor = None
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        _existing_output_is_valid(parent_descriptor, parts[-1], path)
        os.replace(
            temporary_name,
            parts[-1],
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
        published = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published and temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


@contextmanager
def atomic_text_writer(
    path: Path, *, root: Path | None = None, newline: str | None = None
) -> Iterator[TextIO]:
    with _atomic_writer(
        path, root=root, binary=False, newline=newline
    ) as stream:
        yield stream  # type: ignore[misc]


@contextmanager
def atomic_binary_writer(
    path: Path, *, root: Path | None = None
) -> Iterator[BinaryIO]:
    with _atomic_writer(path, root=root, binary=True) as stream:
        yield stream  # type: ignore[misc]


def atomic_write_json(path: Path, payload: Any, *, root: Path | None = None) -> None:
    with atomic_text_writer(path, root=root) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def atomic_write_text(path: Path, text: str, *, root: Path | None = None) -> None:
    with atomic_text_writer(path, root=root) as stream:
        stream.write(text)


def safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "unnamed"


def trusted_system_executable(name: str) -> str:
    """Resolve a non-group/world-writable executable on the system path.

    Benchmark tools are trusted local dependencies, not a privilege boundary.
    Requiring UID 0 made otherwise valid hermetic and container images
    unusable without making backend execution safe against hostile same-UID
    code.  Publication provenance records the resolved path and digest instead.
    """
    candidate = shutil.which(name, path=os.defpath)
    if candidate is None:
        raise OSError(f"trusted system {name} is unavailable")
    path = Path(candidate).resolve(strict=True)
    for component in (path, *path.parents):
        metadata = component.stat(follow_symlinks=False)
        if metadata.st_mode & 0o022:
            raise OSError(
                f"trusted system {name} path is writable by group/other: {component}"
            )
        if component == Path(component.anchor):
            break
    return str(path)


def _command_output(
    cmd: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> str | None:
    output = _command_bytes(cmd, cwd, env=env, pass_fds=pass_fds)
    return os.fsdecode(output).strip() if output is not None else None


def _command_bytes(
    cmd: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> bytes | None:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            pass_fds=pass_fds,
            start_new_session=True,
        )
        if process.stdout is None:
            raise OSError("command stdout pipe is unavailable")
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
        deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
        output = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                return None
            try:
                block = os.read(
                    descriptor,
                    min(
                        _COMMAND_READ_CHUNK_BYTES,
                        MAX_GIT_COMMAND_OUTPUT_BYTES + 1 - len(output),
                    ),
                )
            except BlockingIOError:
                continue
            if not block:
                break
            output.extend(block)
            if len(output) > MAX_GIT_COMMAND_OUTPUT_BYTES:
                return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return None
        return bytes(output) if returncode == 0 else None
    except (OSError, ValueError):
        return None
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if process.stdout is not None:
                process.stdout.close()


def _digest_component(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _git_environment(path: Path) -> dict[str, str]:
    lexical_path = _absolute_lexical(path)
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CEILING_DIRECTORIES": str(lexical_path.parent),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_WORK_TREE": str(lexical_path),
        "HOME": os.sep,
        "LC_ALL": "C",
        "PATH": os.defpath,
    }


def _git_command(git: str, *arguments: str) -> list[str]:
    return [
        git,
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.fileMode=true",
        "-c",
        "core.ignoreCase=false",
        "-c",
        "core.symlinks=true",
        "-c",
        f"core.excludesFile={os.devnull}",
        *arguments,
    ]


def _git_worktree_digest(
    path: Path,
    git: str,
    environment: dict[str, str],
    root_descriptor: int,
) -> str | None:
    entries = _command_bytes(
        _git_command(
            git,
            "ls-files",
            "-z",
            "--full-name",
            "--cached",
            "--others",
            "--exclude-standard",
        ),
        path,
        env=environment,
        pass_fds=(root_descriptor,),
    )
    if entries is None:
        return None

    relative_paths = sorted(entry for entry in entries.split(b"\0") if entry)
    digest = hashlib.sha256()
    digest.update(b"silex-bench-git-worktree-v1\0")
    total_content_bytes = 0
    for relative_bytes in relative_paths:
        parts = tuple(relative_bytes.split(b"/"))
        if (
            relative_bytes.startswith(b"/")
            or any(part in {b"", b".", b".."} for part in parts)
        ):
            raise ValueError("Git returned an unsafe worktree path")
        _digest_component(digest, relative_bytes)
        parent_descriptor = os.dup(root_descriptor)
        leaf_descriptor: int | None = None
        try:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            for component in parts[:-1]:
                child = os.open(component, directory_flags, dir_fd=parent_descriptor)
                os.close(parent_descriptor)
                parent_descriptor = child
            try:
                metadata = os.stat(
                    parts[-1], dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                _digest_component(digest, b"missing")
                continue

            executable = int(bool(metadata.st_mode & 0o111))
            _digest_component(digest, executable.to_bytes(1, byteorder="big"))
            if stat.S_ISLNK(metadata.st_mode):
                _digest_component(digest, b"symlink")
                target = os.readlink(parts[-1], dir_fd=parent_descriptor)
                finished = os.stat(
                    parts[-1], dir_fd=parent_descriptor, follow_symlinks=False
                )
                if _file_signature(metadata) != _file_signature(finished):
                    raise ValueError("Git worktree symlink changed while being hashed")
                total_content_bytes += len(target)
                if total_content_bytes > MAX_GIT_WORKTREE_BYTES:
                    raise ValueError("Git worktree content exceeds the byte limit")
                _digest_component(digest, target)
            elif stat.S_ISREG(metadata.st_mode):
                _digest_component(digest, b"file")
                leaf_descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_descriptor,
                )
                opened = os.fstat(leaf_descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _file_signature(metadata) != _file_signature(opened)
                ):
                    raise ValueError("Git worktree file changed before being hashed")
                total_content_bytes += opened.st_size
                if total_content_bytes > MAX_GIT_WORKTREE_BYTES:
                    raise ValueError("Git worktree content exceeds the byte limit")
                content = hashlib.sha256()
                while block := os.read(leaf_descriptor, 1024 * 1024):
                    content.update(block)
                finished = os.fstat(leaf_descriptor)
                if _file_signature(opened) != _file_signature(finished):
                    raise ValueError("Git worktree file changed while being hashed")
                _digest_component(digest, content.digest())
            elif stat.S_ISDIR(metadata.st_mode):
                # A directory entry from ls-files is normally a gitlink. Its own
                # repository identity is already represented by the parent index.
                _digest_component(digest, b"directory")
            else:
                _digest_component(digest, b"other")
        finally:
            if leaf_descriptor is not None:
                os.close(leaf_descriptor)
            os.close(parent_descriptor)
    return digest.hexdigest()


def _git_index_flag_status(
    path: Path,
    git: str,
    environment: dict[str, str],
    root_descriptor: int,
) -> str | None:
    payload = _command_bytes(
        _git_command(git, "ls-files", "-v", "-z", "--cached"),
        path,
        env=environment,
        pass_fds=(root_descriptor,),
    )
    if payload is None:
        return None
    flagged: list[bytes] = []
    for record in (entry for entry in payload.split(b"\0") if entry):
        if len(record) < 3 or record[1:2] != b" ":
            return None
        tag = record[:1]
        if tag.islower() or tag == b"S":
            flagged.append(record[2:])
    return "\n".join(
        f"!! index flag prevents complete status: {relative!r}"
        for relative in flagged
    )


def git_identity(path: Path) -> dict[str, Any]:
    lexical_path = _absolute_lexical(path)
    root_descriptor: int | None = None
    try:
        git = trusted_system_executable("git")
        root_descriptor = _open_directory_chain(lexical_path, (), create=False)
        git_entry = os.stat(b".git", dir_fd=root_descriptor, follow_symlinks=False)
        if not (
            stat.S_ISDIR(git_entry.st_mode) or stat.S_ISREG(git_entry.st_mode)
        ):
            raise ValueError("repository root .git entry must be a file or directory")
        descriptor_path = Path(f"/proc/self/fd/{root_descriptor}")
        environment = _git_environment(descriptor_path)
        revision = _command_output(
            _git_command(git, "rev-parse", "HEAD"),
            descriptor_path,
            env=environment,
            pass_fds=(root_descriptor,),
        )
        status = _command_output(
            _git_command(git, "status", "--porcelain"),
            descriptor_path,
            env=environment,
            pass_fds=(root_descriptor,),
        )
        index_flag_status = _git_index_flag_status(
            descriptor_path, git, environment, root_descriptor
        )
        worktree_sha256 = _git_worktree_digest(
            descriptor_path, git, environment, root_descriptor
        )
    except (OSError, ValueError):
        revision = None
        status = None
        index_flag_status = None
        worktree_sha256 = None
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
    if status is None or index_flag_status is None or worktree_sha256 is None:
        revision = None
        status = None
        worktree_sha256 = None
    elif index_flag_status:
        status = "\n".join(value for value in (status, index_flag_status) if value)
    return {
        "path": str(lexical_path),
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
        "status": status,
        "worktree_sha256": worktree_sha256,
    }


def machine_identity(cpu: int | None) -> dict[str, Any]:
    cpu_model: str | None = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
        for label in ("model name", "hardware", "cpu model", "model"):
            for line in cpuinfo:
                name, separator, value = line.partition(":")
                if separator and name.strip().lower() == label and value.strip():
                    cpu_model = value.strip()
                    break
            if cpu_model is not None:
                break
    except OSError:
        pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "available_affinity": affinity,
        "requested_cpu": cpu,
    }
