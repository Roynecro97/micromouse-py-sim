"""generic robot utilities

Utility functions for all robots.
"""

from __future__ import annotations

import os
import random

from enum import auto, Enum
from typing import NamedTuple, overload, TYPE_CHECKING

from ..maze import Direction, ExtendedMaze, Maze, RelativeDirection, Walls

if TYPE_CHECKING:
    from collections.abc import Iterable, Generator
    from typing import Callable, Literal

ENABLE_VICTORY_DANCE = os.environ.get('MICROMOUSE_VICTORY_DANCE', 'n') == 'y'

# TODO: replace prints with logging


class Action(Enum):
    """TODO: docs"""
    READY = auto()
    RESET = auto()
    FORWARD = auto()
    BACKWARDS = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()

    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.FRONT]) -> Literal[Action.FORWARD]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.BACK]) -> Literal[Action.BACKWARDS]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.LEFT]) -> Literal[Action.TURN_LEFT]: ...
    @overload
    @classmethod
    def from_rel_direction(cls, direction: Literal[RelativeDirection.RIGHT]) -> Literal[Action.TURN_RIGHT]: ...

    @classmethod
    def from_rel_direction(cls, direction: RelativeDirection) -> Action:
        """Create an action from a relative direction."""
        match direction:
            case RelativeDirection.FRONT: return cls.FORWARD
            case RelativeDirection.BACK: return cls.BACKWARDS
            case RelativeDirection.LEFT: return cls.TURN_LEFT
            case RelativeDirection.RIGHT: return cls.TURN_RIGHT


def turns_for_rel_direction(
    direction: RelativeDirection,
    follow: Literal[RelativeDirection.LEFT, RelativeDirection.RIGHT] = RelativeDirection.LEFT,
) -> list[Literal[Action.TURN_LEFT, Action.TURN_RIGHT]]:
    """Create the turns needed to face in a relative direction."""
    match direction:
        case RelativeDirection.FRONT: return []
        case RelativeDirection.BACK: return [Action.from_rel_direction(follow.invert())] * 2
        case RelativeDirection.LEFT: return [Action.TURN_LEFT]
        case RelativeDirection.RIGHT: return [Action.TURN_RIGHT]


def needed_turns_for_rel_direction(
    direction: RelativeDirection,
    turn_action: Literal[Action.TURN_LEFT, Action.TURN_RIGHT],
) -> int:
    """Get the number of turns needed to face in a relative direction using only one turn type."""
    match direction:
        case RelativeDirection.FRONT: return 0
        case RelativeDirection.BACK: return 2
        case RelativeDirection.LEFT: return 3 if turn_action is Action.TURN_RIGHT else 1
        case RelativeDirection.RIGHT: return 3 if turn_action is Action.TURN_LEFT else 1


class RobotState(NamedTuple):
    """TODO: docs

    Represents the current robot's state
    """
    row: int
    col: int
    facing: Direction


if TYPE_CHECKING:
    type Robot = Generator[Action, RobotState, None]
    type Algorithm = Callable[[ExtendedMaze, set[tuple[int, int]]], Robot]


def _wall_to_direction(wall: Walls) -> Direction:
    match wall:
        case Walls.NORTH | Walls.EAST | Walls.SOUTH | Walls.WEST:
            return Direction(wall.value)
        case _:
            raise ValueError(f"can only convert a single wall (not {wall!r})")


def walls_to_directions(walls: Walls) -> list[Direction]:
    """TODO: docs (converts a cell's wall spec into a sorted list of available directions)"""
    return sorted(_wall_to_direction(missing_wall) for missing_wall in ~walls)


def direction_to_wall(direction: Direction) -> Walls:
    """TODO: docs (converts a direction spec into a wall)"""
    match direction:
        case Direction.NORTH: return Walls.NORTH
        case Direction.EAST: return Walls.EAST
        case Direction.SOUTH: return Walls.SOUTH
        case Direction.WEST: return Walls.WEST
        case _: raise ValueError(f"can only convert the primary directions (not {direction})")


def _adjacent_cells_impl(maze: Maze, cells: Iterable[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    for row, col in cells:
        walls = maze[row, col]
        if Walls.NORTH not in walls:
            yield (row - 1, col)
        if Walls.EAST not in walls:
            yield (row, col + 1)
        if Walls.SOUTH not in walls:
            yield (row + 1, col)
        if Walls.WEST not in walls:
            yield (row, col - 1)


def adjacent_cells(maze: Maze, cells: Iterable[tuple[int, int]], without: set[tuple[int, int]] | None = None) -> set[tuple[int, int]]:
    """
    Returns a cell wil all cells that are adjacent (without diagonals) to a
    cell in ``cells`` and don't have a wall separating them from their relevant
    cell in ``cells``, excluding the ``without`` cells.

    Args:
        maze (Maze): The maze.
        cells (Iterable[tuple[int, int]]): The cells to find adjacent cells of.
        without (set[tuple[int, int]] | None, optional): Cells to exclude from the result. Defaults to None.

    Returns:
        set[tuple[int, int]]: A cell with all relevant adjacent cells.
    """
    return set(_adjacent_cells_impl(maze, cells)) - (without or set())


def direction_to_cell(cell: tuple[int, int], direction: Direction) -> tuple[int, int]:
    """
    Returns the indexes of the cell to the ``direction`` of the given cell.

    Args:
        cell (tuple[int, int]): The current cell.
        direction (Direction): The direction to move at. Must be a primary direction.

    Raises:
        ValueError: ``direction`` is a secondary direction.

    Returns:
        tuple[int, int]: The required cell, may be out of bounds of the maze.
    """
    row, col = cell
    match direction:
        case Direction.NORTH:
            return (row - 1, col)
        case Direction.EAST:
            return (row, col + 1)
        case Direction.SOUTH:
            return (row + 1, col)
        case Direction.WEST:
            return (row, col - 1)
    raise ValueError(f"unsupported direction {direction}")



def identity[T](obj: T) -> T:
    """Return ``obj``.

    Args:
        obj (T): Any python object.

    Returns:
        T: ``obj``.

    >>> x = [1, 2, 3]
    >>> identity(x) is x
    True

    >>> x = object()
    >>> identity(x) is x
    True
    """
    return obj


def shuffled[T](lst: list[T]) -> list[T]:
    """Shuffle the list (in-place) and return it.

    Args:
        lst (list[T]): The list to shuffle.

    Returns:
        list[T]: ``lst`` after shuffling.

    >>> lst = [1, 2, 3]
    >>> shuffled(lst) is lst
    True
    """
    random.shuffle(lst)
    return lst
