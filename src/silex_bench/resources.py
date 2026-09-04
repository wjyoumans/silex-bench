"""Discovery and path resolution for installed benchmark resources."""

from __future__ import annotations

from importlib.resources import files
from os import PathLike
from pathlib import Path


_SUFFIXES = {
    "suites": ".toml",
    "profiles": ".toml",
    "corpora": ".json",
}


def _suffix(collection: str) -> str:
    try:
        return _SUFFIXES[collection]
    except KeyError as exc:
        choices = ", ".join(sorted(_SUFFIXES))
        raise ValueError(
            f"unknown resource collection {collection!r}; expected one of: {choices}"
        ) from exc


def resource_root() -> Path:
    """Return the package-data root for a normal unpacked installation."""

    traversable = files("silex_bench").joinpath("data")
    try:
        root = Path(traversable)
    except TypeError as exc:  # pragma: no cover - normal wheels install unpacked
        raise RuntimeError(
            "silex-bench resources require a normal unpacked installation"
        ) from exc
    if not root.is_dir():
        raise RuntimeError(f"installed silex-bench resources are missing: {root}")
    return root.absolute()


def _plain_name(name: str | PathLike[str]) -> str:
    path = Path(name)
    if not path.name or path.parent != Path(".") or path.name in {".", ".."}:
        raise ValueError(f"expected a plain resource name, got {str(name)!r}")
    return path.name


def builtin_names(collection: str) -> tuple[str, ...]:
    """List the stable names of resources in a built-in collection."""

    suffix = _suffix(collection)
    directory = resource_root() / collection
    if not directory.is_dir():
        raise RuntimeError(f"installed resource collection is missing: {directory}")
    return tuple(
        sorted(
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix == suffix
        )
    )


def builtin_path(collection: str, name: str | PathLike[str]) -> Path:
    """Resolve one built-in resource name to its installed filesystem path."""

    suffix = _suffix(collection)
    filename = _plain_name(name)
    if not filename.endswith(suffix):
        filename += suffix
    path = (resource_root() / collection / filename).absolute()
    if not path.is_file():
        choices = ", ".join(builtin_names(collection))
        raise FileNotFoundError(
            f"unknown built-in {collection[:-1]} {Path(filename).stem!r}; "
            f"available: {choices or '(none)'}"
        )
    return path


def resolve_path(collection: str, value: str | PathLike[str]) -> Path:
    """Resolve a built-in name or preserve an explicit custom resource path."""

    _suffix(collection)
    supplied = Path(value).expanduser()
    if supplied.is_file() or supplied.is_absolute() or supplied.parent != Path("."):
        return supplied.absolute()
    return builtin_path(collection, supplied.name)
