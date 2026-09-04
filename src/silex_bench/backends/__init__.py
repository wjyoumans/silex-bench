"""Backend adapter registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BackendAdapter


def create_backend(name: str) -> "BackendAdapter":
    if name == "silex":
        from .silex import SilexBackend

        return SilexBackend()
    if name == "pari":
        from .pari import PariBackend

        return PariBackend()
    if name == "hecke":
        from .hecke import HeckeBackend

        return HeckeBackend()
    if name == "magma":
        from .magma import MagmaBackend

        return MagmaBackend()
    raise ValueError(f"unknown backend: {name}")
