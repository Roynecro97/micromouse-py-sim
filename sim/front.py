"""Utilities for using frontends this package.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulator import Simulator


class Renderer(ABC):  # pylint: disable=too-few-public-methods
    """Base renderer class for simulation (GUI) renderers."""

    def __init__(self, sim: Simulator) -> None:
        super().__init__()
        self.sim = sim

    @abstractmethod
    def run(self) -> None:
        """Start the renderer."""
        raise NotImplementedError
